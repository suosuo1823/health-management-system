// =============================================
//  健康管理平台 - 全局 JavaScript
// =============================================

// ---- Toast 通知 ----
window.HP = window.HP || {};

HP.toast = function(msg, type = 'success') {
  const colors = {
    success: '#2d8653',
    danger:  '#dc3545',
    warning: '#f0a500',
    info:    '#1a73e8',
  };
  const icons = {
    success: 'fa-check-circle',
    danger:  'fa-times-circle',
    warning: 'fa-exclamation-triangle',
    info:    'fa-info-circle',
  };
  let container = document.getElementById('hp-toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'hp-toast-container';
    container.className = 'toast-container';
    document.body.appendChild(container);
  }
  const el = document.createElement('div');
  el.className = 'hp-toast';
  el.style.borderLeftColor = colors[type] || colors.success;
  el.innerHTML = `
    <div style="display:flex;align-items:center;gap:10px;">
      <i class="fa ${icons[type] || icons.success}" style="color:${colors[type]};font-size:16px;"></i>
      <span style="flex:1;">${msg}</span>
      <button onclick="this.closest('.hp-toast').remove()" style="border:none;background:none;cursor:pointer;font-size:16px;color:#999;line-height:1;">&times;</button>
    </div>`;
  container.appendChild(el);
  setTimeout(() => { if (el.parentNode) el.remove(); }, 4000);
};

// ---- 移动端侧栏开关 ----
document.addEventListener('DOMContentLoaded', function () {
  const toggleBtn = document.getElementById('sidebar-toggle');
  const sidebar   = document.getElementById('app-sidebar');
  const overlay   = document.getElementById('sidebar-overlay');

  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener('click', function () {
      sidebar.classList.toggle('open');
      if (overlay) overlay.classList.toggle('d-none');
    });
  }
  if (overlay && sidebar) {
    overlay.addEventListener('click', function () {
      sidebar.classList.remove('open');
      overlay.classList.add('d-none');
    });
  }

  // 高亮当前菜单
  const currentPath = window.location.pathname;
  document.querySelectorAll('.sidebar-link').forEach(link => {
    const href = link.getAttribute('href');
    if (href && currentPath.startsWith(href) && href !== '/') {
      link.classList.add('active');
    } else if (href === '/' && currentPath === '/') {
      link.classList.add('active');
    }
  });

  // Flash 消息自动淡出
  document.querySelectorAll('.alert-dismissible').forEach(function(el) {
    setTimeout(function() {
      el.classList.add('fade');
      setTimeout(() => el.remove(), 500);
    }, 4000);
  });
});

// ---- 通用 fetch 封装 ----
HP.post = async function(url, data) {
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
    body: JSON.stringify(data),
  });
  return resp.json();
};

HP.delete = async function(url) {
  const resp = await fetch(url, {
    method: 'DELETE',
    headers: { 'X-Requested-With': 'XMLHttpRequest' },
  });
  return resp.json();
};

// ---- 数字格式化 ----
HP.fmt = function(n, decimals = 1) {
  return parseFloat(n).toFixed(decimals);
};

// ---- 日期工具 ----
HP.today = function() {
  return new Date().toISOString().slice(0, 10);
};

// ---- ECharts 通用主题配置 ----
HP.chartTheme = {
  color: ['#2d8653','#52c97f','#1a73e8','#f0a500','#dc3545','#7b2ff7','#17a2b8'],
  textStyle:  { color: '#1c3028', fontFamily: "'Segoe UI','PingFang SC',sans-serif" },
  grid:       { top: 40, right: 20, bottom: 40, left: 50, containLabel: true },
  tooltip:    { backgroundColor: '#fff', borderColor: '#d8edd9', borderWidth: 1,
                textStyle: { color: '#1c3028', fontSize: 13 } },
  legend:     { textStyle: { color: '#6b8a77', fontSize: 12 } },
  axisLine:   { lineStyle: { color: '#d8edd9' } },
  axisLabel:  { color: '#6b8a77', fontSize: 12 },
  splitLine:  { lineStyle: { color: '#edf7f0', type: 'dashed' } },
};

HP.darkChartTheme = {
  color: ['#52c97f','#6de095','#1a73e8','#f0a500','#dc3545','#7b2ff7','#17a2b8'],
  textStyle:  { color: '#e0f0e6', fontFamily: "'Segoe UI','PingFang SC',sans-serif" },
  grid:       { top: 40, right: 20, bottom: 40, left: 50, containLabel: true },
  tooltip:    { backgroundColor: '#152a1e', borderColor: '#1e3d2a', borderWidth: 1,
                textStyle: { color: '#e0f0e6', fontSize: 13 } },
  legend:     { textStyle: { color: '#7aa38d', fontSize: 12 } },
  axisLine:   { lineStyle: { color: '#1e3d2a' } },
  axisLabel:  { color: '#7aa38d', fontSize: 12 },
  splitLine:  { lineStyle: { color: '#1e3d2a', type: 'dashed' } },
};

// 获取当前图表主题
HP.getChartTheme = function() {
  return document.documentElement.getAttribute('data-theme') === 'dark'
    ? HP.darkChartTheme : HP.chartTheme;
};

// ---- 亮/暗主题切换 ----
HP.setTheme = function(mode) {
  var html = document.documentElement;
  if (mode === 'dark') {
    html.setAttribute('data-theme', 'dark');
    localStorage.setItem('hp-theme', 'dark');
  } else if (mode === 'light') {
    html.removeAttribute('data-theme');
    localStorage.setItem('hp-theme', 'light');
  } else {
    html.removeAttribute('data-theme');
    localStorage.setItem('hp-theme', 'system');
  }
  HP.updateThemeBtn();
};

HP.updateThemeBtn = function() {
  var t = localStorage.getItem('hp-theme') || 'system';
  var isDark = t === 'dark' || (t === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
  var icon = document.getElementById('themeIcon');
  var label = document.getElementById('themeLabel');
  if (icon) {
    icon.className = 'fa ' + (isDark ? 'fa-sun-o' : 'fa-moon-o');
  }
  if (label) {
    label.textContent = isDark ? '浅色' : '深色';
  }
};

// 监听系统主题变化
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
  if (localStorage.getItem('hp-theme') === 'system') {
    if (e.matches) {
      document.documentElement.setAttribute('data-theme', 'dark');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
    HP.updateThemeBtn();
  }
});

// 主题切换按钮绑定
document.addEventListener('DOMContentLoaded', function() {
  HP.updateThemeBtn();
  var btn = document.getElementById('themeToggleBtn');
  if (btn) {
    btn.addEventListener('click', function() {
      var t = localStorage.getItem('hp-theme') || 'system';
      if (t === 'dark' || (t === 'system' && document.documentElement.getAttribute('data-theme'))) {
        HP.setTheme('light');
      } else {
        HP.setTheme('dark');
      }
    });
  }
});
