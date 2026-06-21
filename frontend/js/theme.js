// Theme toggle & Profile sync logic
(function() {
    // Run immediately to prevent FOUC (Flash of Unstyled Content)
    const storedTheme = localStorage.getItem('theme');
    if (storedTheme === 'light') {
        document.documentElement.classList.add('light-mode');
    } else {
        document.documentElement.classList.remove('light-mode');
    }
})();

document.addEventListener('DOMContentLoaded', () => {
    // 1. Theme Toggler Logic
    const themeToggler = document.getElementById('theme-toggler');
    if (themeToggler) {
        const icon = themeToggler.querySelector('i');
        
        function updateIcon() {
            if (!icon) return;
            if (document.documentElement.classList.contains('light-mode')) {
                icon.classList.remove('fa-moon');
                icon.classList.add('fa-sun');
            } else {
                icon.classList.remove('fa-sun');
                icon.classList.add('fa-moon');
            }
        }
        
        updateIcon();
        
        themeToggler.addEventListener('click', () => {
            document.documentElement.classList.toggle('light-mode');
            const isLight = document.documentElement.classList.contains('light-mode');
            localStorage.setItem('theme', isLight ? 'light' : 'dark');
            updateIcon();
        });
    }

    // 2. Global Profile Sync Logic
    function syncProfile() {
        const savedName = localStorage.getItem('userName') || 'Simran';
        const sidebarName = document.getElementById('sidebar-name');
        const globalAvatar = document.getElementById('global-avatar');
        
        if (sidebarName) {
            sidebarName.innerHTML = `${savedName}<span class="highlight">AI</span>`;
        }
        if (globalAvatar) {
            globalAvatar.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(savedName)}&background=0D8ABC&color=fff`;
        }
    }

    syncProfile();
    
    // 3. Global Logout Logic - Removed
    
    // 4. Auth Gate - Removed

    // Listen for storage changes in other tabs
    window.addEventListener('storage', (e) => {
        if (e.key === 'userName' || e.key === 'theme') {
            syncProfile();
            if (e.key === 'theme') {
                if (e.newValue === 'light') document.documentElement.classList.add('light-mode');
                else document.documentElement.classList.remove('light-mode');
            }
        }
    });
});
