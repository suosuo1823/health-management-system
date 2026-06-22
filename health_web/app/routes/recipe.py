# -*- coding: utf-8 -*-
"""
app/routes/recipe.py  -  菜谱库路由

Bug 修复清单：
  1. _nutrition_value 调用时 nut_map=None 导致崩溃 → api_detail 中单独从 nutrients 列表取值
  2. Recipe.calories 是 Python property 不是列，不能用于 ORDER BY → 改为子查询排序
  3. lazy="dynamic" + joinedload 冲突 → model 改为 lazy="select" 后此处 joinedload 生效
"""
from flask import Blueprint, request, jsonify, render_template
from flask_login import current_user, login_required
from sqlalchemy.orm import joinedload
from sqlalchemy import func as sql_func, case
from app import db
from app.models.recipe import Recipe, RecipeNutrition
from app.models.diet import DietRecord
from datetime import date

bp = Blueprint("recipe", __name__, url_prefix="/recipes")

# ── 常量 ─────────────────────────────────────────────────────────────
NUTRIENT_LABELS = {
    "能量":    "热量(kcal)", "蛋白质": "蛋白质(g)",
    "脂肪":    "脂肪(g)",   "碳水化合物": "碳水(g)",
    "膳食纤维": "膳食纤维(g)",
    "维生素A":  "维生素A",   "维生素C": "维生素C",
    "维生素E":  "维生素E",   "胡萝卜素": "胡萝卜素",
    "硫胺素":   "硫胺素",    "核黄素":   "核黄素",
    "烟酸":     "烟酸",      "胆固醇":   "胆固醇",
    "钾": "钾", "钠": "钠",  "钙": "钙",  "镁": "镁",
    "铁": "铁", "锰": "锰",  "锌": "锌",  "铜": "铜",
    "磷": "磷", "硒": "硒",
}
MEAL_LABELS = {
    "breakfast": "早餐", "lunch": "午餐",
    "dinner": "晚餐",   "snack": "加餐",
}
# 内存缓存：筛选选项（只在首次或强制刷新时查数据库）
_FILTER_CACHE: dict = {"gongyi": [], "kouwei": [], "loaded": False}


# ── 工具函数 ──────────────────────────────────────────────────────────

def _parse_float(val: str | None) -> float | None:
    """把带单位的字符串解析成 float，失败返回 None"""
    if not val:
        return None
    s = str(val).strip()
    for unit in ("千卡", "kcal", "克", "g", "毫克", "mg",
                 "微克", "μg", "μg"):
        s = s.replace(unit, "")
    try:
        return float(s.strip())
    except (ValueError, TypeError):
        return None


def _build_nutrition_map(recipe_ids: list) -> dict:
    """
    批量查询一批 recipe_id 的全部营养数据。
    返回 {recipe_id: {nutrient_key: float_or_None}}
    一次 SQL，零 N+1。
    """
    if not recipe_ids:
        return {}
    rows = db.session.query(
        RecipeNutrition.recipe_id,
        RecipeNutrition.nutrient_key,
        RecipeNutrition.nutrient_value,
    ).filter(RecipeNutrition.recipe_id.in_(recipe_ids)).all()

    mapping: dict = {}
    for rid, key, value in rows:
        mapping.setdefault(rid, {})[key] = _parse_float(value)
    return mapping


def _compute_recommend_score(nd: dict) -> float:
    """
    根据实际营养数据计算健康推荐指数（0~10）。
    算法逻辑：
      - 蛋白密度得分：蛋白质/热量比值越高越好（优质蛋白来源）
      - 纤维加分：富含膳食纤维有助于饱腹和肠道健康
      - 热量惩罚：热量过高不利于体重控制
      - 脂肪惩罚：脂肪过高增加热量负担
      - 低碳水奖励：适当控制碳水有利于健康饮食
    """
    cal   = nd.get("能量")         # kcal
    prot  = nd.get("蛋白质")       # g
    fat   = nd.get("脂肪")          # g
    carbs = nd.get("碳水化合物")   # g
    fiber = nd.get("膳食纤维")     # g

    # 缺少核心数据时给中等偏低分
    if cal is None or prot is None:
        return 5.0

    cal   = float(cal)
    prot  = float(prot)
    fat   = float(fat) if fat else 0.0
    carbs = float(carbs) if carbs else 0.0
    fiber = float(fiber) if fiber else 0.0

    total_macros = prot + fat + carbs
    if total_macros <= 0:
        return 5.0

    # 蛋白密度：蛋白质占总供能物质的比例（越高越优质）
    protein_ratio = prot / total_macros          # 0~1
    protein_density = prot / (cal + 1) * 100     # g/100kcal，越高越好

    # 纤维加分（上限 +1.5）
    fiber_score = min(1.5, fiber * 0.3)

    # 热量惩罚（每50kcal扣0.5分，上限-3）
    cal_penalty = min(3.0, cal / 50.0 * 0.5)

    # 脂肪惩罚（每10g扣0.5分，上限-2）
    fat_penalty = min(2.0, fat / 10.0 * 0.5)

    # 蛋白密度基础分（0~4分，密度5g/100kcal以上给满分）
    prot_score = min(4.0, protein_density / 5.0 * 4.0)

    # 蛋白比例加分（0~2分，>20%给满分）
    prot_ratio_score = min(2.0, protein_ratio / 0.20 * 2.0)

    # 低碳水奖励（<15%给+1分）
    carb_ratio = carbs / (total_macros + 0.001)
    carb_bonus = 1.0 if carb_ratio < 0.15 else 0.0

    score = prot_score + prot_ratio_score + fiber_score - cal_penalty - fat_penalty + carb_bonus

    # 限制在 0~10 范围
    return max(0.0, min(10.0, round(score, 1)))


def _recipe_to_card(r: Recipe, nut_map: dict) -> dict:
    """ORM → 列表卡片字典（营养从 nut_map 取，不触发懒加载）"""
    nd = nut_map.get(r.id, {})
    return {
        "id":              r.id,
        "title":           r.title,
        "cooking_method":  r.cooking_method,
        "taste":           r.taste,
        "category":        r.category,
        "main_ingredient": r.main_ingredient,
        "calories":        nd.get("能量"),
        "protein":         nd.get("蛋白质"),
        "fat":             nd.get("脂肪"),
        "carbs":           nd.get("碳水化合物"),
        "recommend_score": _compute_recommend_score(nd),
    }


def _load_filter_cache():
    """填充内存缓存（仅首次调用时触发 SQL）"""
    if _FILTER_CACHE["loaded"]:
        return
    gongyi = sorted(set(
        r[0] for r in db.session.query(Recipe.cooking_method).all() if r[0]
    ))
    kouwei = sorted(set(
        r[0] for r in db.session.query(Recipe.taste).all() if r[0]
    ))
    _FILTER_CACHE["gongyi"] = gongyi
    _FILTER_CACHE["kouwei"] = kouwei
    _FILTER_CACHE["loaded"] = True


# ── 路由 ──────────────────────────────────────────────────────────────

@bp.route("")
def index():
    return render_template("recipe/index.html")


@bp.route("/api/filters")
def api_filters():
    """返回工艺/口味筛选选项（内存缓存，重启前只查一次数据库）"""
    _load_filter_cache()
    return jsonify({
        "gongyi_opts": _FILTER_CACHE["gongyi"],
        "kouwei_opts": _FILTER_CACHE["kouwei"],
    })


@bp.route("/api/list")
def api_list():
    """
    菜谱分页列表 API

    Query params:
      q           – 关键字（菜名/主料/类别模糊搜索）
      gongyi      – 烹饪工艺精确筛选
      kouwei      – 口味精确筛选
      sort        – recommend（默认）| calories_asc | calories_desc
      page        – 页码（≥1）
      per_page    – 每页条数（6~48，默认12）
    """
    q        = request.args.get("q", "").strip()
    gongyi   = request.args.get("gongyi", "").strip()
    kouwei   = request.args.get("kouwei", "").strip()
    sort_key = request.args.get("sort", "recommend")
    page     = max(1, int(request.args.get("page", 1) or 1))
    per_page = min(48, max(6, int(request.args.get("per_page", 12) or 12)))

    # ─── 构建基础查询（只查主表，不 join 营养，营养后面批量取）───
    query = db.session.query(Recipe)

    if q:
        pat = f"%{q}%"
        query = query.filter(db.or_(
            Recipe.title.ilike(pat),
            Recipe.main_ingredient.ilike(pat),
            Recipe.category.ilike(pat),
        ))
    if gongyi:
        query = query.filter(Recipe.cooking_method == gongyi)
    if kouwei:
        query = query.filter(Recipe.taste == kouwei)

    # ─── 排序 ───────────────────────────────────────────────────
    # calories_asc/desc 需要通过子查询 JOIN 到营养表才能在 SQL 里排序
    if sort_key in ("calories_asc", "calories_desc"):
        # 子查询：每道菜的能量值（数值化）
        cal_sub = (
            db.session.query(
                RecipeNutrition.recipe_id,
                sql_func.max(RecipeNutrition.nutrient_value).label("cal_str"),
            )
            .filter(RecipeNutrition.nutrient_key == "能量")
            .group_by(RecipeNutrition.recipe_id)
            .subquery()
        )
        query = query.outerjoin(cal_sub, Recipe.id == cal_sub.c.recipe_id)
        if sort_key == "calories_asc":
            query = query.order_by(
                cal_sub.c.cal_str.is_(None),   # NULL 排最后
                cal_sub.c.cal_str.asc(),
            )
        else:
            query = query.order_by(
                cal_sub.c.cal_str.is_(None),
                cal_sub.c.cal_str.desc(),
            )
    else:
        # 默认按推荐分降序（MySQL 不支持 NULLS LAST，用 CASE 模拟）
        query = query.order_by(
            case((Recipe.recommend_score.is_(None), 1), else_=0),
            Recipe.recommend_score.desc(),
        )

    # ─── 分页 ───────────────────────────────────────────────────
    total = query.count()
    rows  = query.offset((page - 1) * per_page).limit(per_page).all()

    # ─── 批量取营养（1 次 SQL）─────────────────────────────────
    nut_map = _build_nutrition_map([r.id for r in rows])
    items   = [_recipe_to_card(r, nut_map) for r in rows]

    return jsonify({
        "items":    items,
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "pages":    max(1, (total + per_page - 1) // per_page),
    })


@bp.route("/api/detail/<int:recipe_id>")
def api_detail(recipe_id):
    """菜谱完整详情 JSON（含全部营养数据），弹窗专用"""
    # joinedload 一次加载全部 nutrients（lazy="select" 模式下生效）
    recipe = (
        db.session.query(Recipe)
        .options(joinedload(Recipe.nutrients))
        .filter(Recipe.id == recipe_id)
        .first_or_404()
    )

    # 直接从已 joinedload 的 nutrients 列表构建字典 → 0 额外 SQL
    all_nutrients: dict = {}
    cal_raw = None
    for n in recipe.nutrients:
        label = NUTRIENT_LABELS.get(n.nutrient_key, n.nutrient_key)
        all_nutrients[label] = n.nutrient_value
        if n.nutrient_key == "能量":
            cal_raw = _parse_float(n.nutrient_value)

    return jsonify({
        "id":               recipe.id,
        "title":            recipe.title,
        "cooking_method":   recipe.cooking_method,
        "taste":            recipe.taste,
        "category":         recipe.category,
        "main_ingredient":  recipe.main_ingredient,
        "sub_ingredient":   recipe.sub_ingredient,
        "seasoning":        recipe.seasoning,
        "instructions":     recipe.instructions,
        "suitable":         recipe.suitable,
        "mouthfeel":        recipe.mouthfeel,
        "all_nutrients":    all_nutrients,
        "recommend_score":  recipe.recommend_score or 0,
        "spicy_score":      recipe.spicy_score      or 0,
        "nutrition_score":  recipe.nutrition_score   or 0,
        "difficulty_score": recipe.difficulty_score  or 0,
        "time_score":       recipe.time_score        or 0,
        "diet_score":       recipe.diet_score        or 0,
        "seasoning_score":  recipe.seasoning_score   or 0,
        "calories":         cal_raw,
    })


@bp.route("/add-to-diet", methods=["POST"])
@login_required
def add_to_diet():
    """将菜谱加入今日饮食记录"""
    recipe_id = request.form.get("recipe_id", type=int)
    meal_type = request.form.get("meal_type", "breakfast")
    portion_g = request.form.get("portion_g", type=float, default=100.0)

    if not recipe_id:
        return jsonify(success=False, message="缺少菜谱ID")

    recipe = db.session.get(Recipe, recipe_id)
    if not recipe:
        return jsonify(success=False, message="菜谱不存在")

    # 批量取营养（1 次 SQL）
    nut_dict = _build_nutrition_map([recipe_id]).get(recipe_id, {})
    ratio = (portion_g or 100.0) / 100.0

    from app.utils import db_transaction
    
    try:
        with db_transaction():
            record = DietRecord(
                user_id=current_user.id,
                food_name=recipe.title,
                meal_type=meal_type,
                portion=f"{int(portion_g)}g",
                calories=round((nut_dict.get("能量")    or 0.0) * ratio, 1),
                protein= round((nut_dict.get("蛋白质")  or 0.0) * ratio, 1),
                carbs=   round((nut_dict.get("碳水化合物") or 0.0) * ratio, 1),
                fat=     round((nut_dict.get("脂肪")    or 0.0) * ratio, 1),
                record_date=date.today(),
            )
            db.session.add(record)
    except Exception as e:
        return jsonify(success=False, message=f"添加失败: {str(e)}")

    cal_str = f"{round((nut_dict.get('能量') or 0) * ratio, 1)} kcal"
    return jsonify(
        success=True,
        message=(
            f"已将「{recipe.title}」({int(portion_g)}g) "
            f"加入今日{MEAL_LABELS.get(meal_type, meal_type)}，"
            f"热量约 {cal_str}"
        ),
    )
