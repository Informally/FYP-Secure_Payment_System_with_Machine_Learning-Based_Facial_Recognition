<?php
// Include admin config
require_once 'config/admin_config.php';

// Get session info before clearing - match your login.php exactly
$session_token = $_SESSION['admin_session_token'] ?? null;
$admin_id = null;

// Get admin_id from session data
if (isset($_SESSION['admin_user']) && is_array($_SESSION['admin_user'])) {
    $admin_id = $_SESSION['admin_user']['id'] ?? null;
}

// Use AdminAuth logout method if session token exists
if ($session_token) {
    $auth = getAdminAuth();
    $auth->logout($session_token, $admin_id);
}

// Clear specific session variables first
unset($_SESSION['admin_session_token']);
unset($_SESSION['admin_user']);

// Clear all session data
$_SESSION = array();

// Delete the session cookie if it exists
if (ini_get("session.use_cookies")) {
    $params = session_get_cookie_params();
    setcookie(session_name(), '', time() - 42000,
        $params["path"], $params["domain"],
        $params["secure"], $params["httponly"]
    );
}

// Clear admin-specific cookies
if (isset($_COOKIE['admin_session'])) {
    setcookie('admin_session', '', time() - 3600, '/admin_side/', '', true, true);
}

// Destroy the session
session_destroy();

// Prevent caching
header("Cache-Control: no-cache, no-store, must-revalidate");
header("Pragma: no-cache");
header("Expires: 0");

// Redirect directly to login
header("Location: login.php");
exit();
?>