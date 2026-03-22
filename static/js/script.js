/* ═══════════════════════════════════════════════════════════
   Telangana Electricity Dashboard – Shared JavaScript
   ═══════════════════════════════════════════════════════════ */

// Close sidebar on mobile when a nav link is clicked
document.querySelectorAll('.sidebar .nav-link').forEach(link => {
    link.addEventListener('click', () => {
        const sidebar = document.getElementById('sidebar');
        if (sidebar && window.innerWidth <= 768) {
            sidebar.classList.remove('open');
        }
    });
});
