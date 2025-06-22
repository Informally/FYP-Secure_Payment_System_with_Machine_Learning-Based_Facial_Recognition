<?php
// admin_side/config/admin_config.php
date_default_timezone_set('Asia/Kuala_Lumpur'); // or your timezone
// Include the AdminAuth class
require_once __DIR__ . '/../classes/AdminAuth.php';

// Database connection - DO NOT load customer config
function dbConnect() {
    try {
        $conn = new mysqli('localhost', 'root', '', 'payment_facial');
        
        if ($conn->connect_error) {
            throw new Exception("Connection failed: " . $conn->connect_error);
        }
        
        $conn->set_charset("utf8mb4");
        return $conn;
        
    } catch (Exception $e) {
        error_log("Database connection failed: " . $e->getMessage());
        die("Database connection failed. Please try again later.");
    }
}

// Add encryption functions (from customer config)
function aes_encrypt($plaintext) {
    $key = base64_decode('0KvocVvCxtHUCXBQ8LVuztXHBFy7fiyjzhxOsXxQCmU=');
    $iv = base64_decode('0OUi5xYbQkZgId+yqOrCUQ==');
    return base64_encode(openssl_encrypt(
        $plaintext,
        'AES-256-CBC',
        $key,
        OPENSSL_RAW_DATA,
        $iv
    ));
}

function aes_decrypt($ciphertext) {
    $key = base64_decode('0KvocVvCxtHUCXBQ8LVuztXHBFy7fiyjzhxOsXxQCmU=');
    $iv = base64_decode('0OUi5xYbQkZgId+yqOrCUQ==');
    return openssl_decrypt(
        base64_decode($ciphertext),
        'AES-256-CBC',
        $key,
        OPENSSL_RAW_DATA,
        $iv
    );
}

function getMerchantName($merchant_id) {
    $conn = dbConnect();
    $stmt = $conn->prepare("SELECT name FROM merchants WHERE merchant_id = ?");
    $stmt->bind_param("s", $merchant_id);
    $stmt->execute();
    $result = $stmt->get_result();
    $conn->close();
    return $result->num_rows > 0 ? $result->fetch_assoc()['name'] : 'Unknown Merchant';
}

// Admin-specific configuration
define('ADMIN_SESSION_DURATION', 8 * 3600); // 8 hours
define('ADMIN_INACTIVITY_TIMEOUT', 2 * 3600); // 2 hours

// Role permissions
$ADMIN_PERMISSIONS = [
    'super_admin' => ['*'], // All permissions
    'financial_admin' => ['view_transactions', 'manage_wallets', 'generate_reports'],
    'kiosk_admin' => ['manage_kiosks', 'manage_products', 'view_transactions'],
    'support_admin' => ['view_users', 'manage_users', 'view_transactions'],
    'audit_admin' => ['view_audit_log', 'view_reports', 'view_transactions']
];

// Initialize admin authentication
$conn = dbConnect();
$admin_auth = new AdminAuth($conn);

/**
 * Get admin authentication instance
 */
function getAdminAuth() {
    global $admin_auth;
    return $admin_auth;
}

/**
 * Check admin session middleware
 */
function requireAdminAuth($required_roles = []) {
    // Debug logging
    error_log("🔍 requireAdminAuth() called");
    
    $auth = getAdminAuth();
    
    // Check for session token
    $session_token = $_SESSION['admin_session_token'] ?? null;
    
    if (!$session_token) {
        error_log("❌ No session token found");
        header("Location: login.php");
        exit();
    }
    
    error_log("✅ Session token found: " . substr($session_token, 0, 20));
    
    // Validate session
    $admin = $auth->validateSession($session_token);
    if (!$admin) {
        error_log("❌ Session validation failed");
        header("Location: login.php");
        exit();
    }
    
    error_log("✅ Session validated for: " . $admin['username']);
    
    // Check role permissions
    if (!empty($required_roles) && $admin['role'] !== 'super_admin') {
        if (!in_array($admin['role'], $required_roles)) {
            error_log("❌ Access denied for role: " . $admin['role']);
            http_response_code(403);
            die("Access denied: Insufficient permissions");
        }
    }
    
    return $admin;
}

// Set unique session name for admin
session_name('FACEPAY_ADMIN_SESSION');

// Initialize session for admin (separate from customer)
if (session_status() === PHP_SESSION_NONE) {
    session_set_cookie_params([
        'lifetime' => ADMIN_SESSION_DURATION,
        'path' => '/',  // Use root path for now
        'secure' => isset($_SERVER['HTTPS']) && $_SERVER['HTTPS'] === 'on',
        'httponly' => true,
        'samesite' => 'Strict'
    ]);
    session_start();
}
?>