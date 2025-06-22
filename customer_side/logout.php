<?php
// logout.php - Secure logout with session cleanup (MySQLi version)
require_once 'config.php';

// Start session to access session data
SessionManager::start();

// Log the logout event
if (isset($_SESSION['authenticatedUserId'])) {
    SecurityUtils::logSecurityEvent(
        $_SESSION['authenticatedUserId'],
        'logout',
        'success',
        [
            'session_duration' => isset($_SESSION['auth_time']) ? (time() - $_SESSION['auth_time']) : 0,
            'logout_method' => $_SERVER['REQUEST_METHOD']
        ]
    );
}

// Clear remember token if exists
if (isset($_COOKIE['remember_token'])) {
    try {
        $conn = dbConnect(); // Use MySQLi connection
        $token_hash = hash('sha256', $_COOKIE['remember_token']);
        
        // Check if remember_tokens table exists first (same as login.php pattern)
        $check_table = $conn->query("SHOW TABLES LIKE 'remember_tokens'");
        if ($check_table->num_rows > 0) {
            // Remove token from database (MySQLi style)
            $stmt = $conn->prepare("DELETE FROM remember_tokens WHERE token = ?");
            $stmt->bind_param("s", $token_hash);
            $stmt->execute();
            $stmt->close();
        }
        
        // Clear cookie
        setcookie('remember_token', '', time() - 3600, '/', '', true, true);
        
        $conn->close();
    } catch (Exception $e) {
        error_log("Error clearing remember token: " . $e->getMessage());
    }
}

// Destroy session
SessionManager::destroy();

// Send JSON response for AJAX requests
if ($_SERVER['REQUEST_METHOD'] === 'POST' &&
    (isset($_SERVER['HTTP_ACCEPT']) && strpos($_SERVER['HTTP_ACCEPT'], 'application/json') !== false)) {
    header('Content-Type: application/json');
    echo json_encode(['status' => 'success', 'message' => 'Logged out successfully']);
    exit();
}

// Redirect for regular requests
header("Location: login.php?logged_out=1");
exit();
?>