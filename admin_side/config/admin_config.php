<?php
/**
 * SECURE ADMIN CONFIG - admin_side/config/admin_config.php
 * Enhanced with Security Functions - Uses shared secure key infrastructure
 */

date_default_timezone_set('Asia/Kuala_Lumpur');
require_once __DIR__ . '/../classes/AdminAuth.php';

// Shared Secure Key Manager (same as customer side)
class AdminSecureKeyManager {
    private static $keyFile = null;
    private static $keys = null;
    
    private static function getKeyFilePath() {
        if (self::$keyFile === null) {
            // Current path: admin_side/config/admin_config.php
            // Target: FYP/secure_keys.json (created by customer_side)
            
            $possiblePaths = [
                // From admin_side/config/ to FYP/
                dirname(dirname(__DIR__)) . '/secure_keys.json',        // ../../secure_keys.json
                
                // Alternative paths if structure differs
                dirname(dirname(dirname(__FILE__))) . '/secure_keys.json',  // Go up 3 levels to FYP/
                __DIR__ . '/../../secure_keys.json',                    // Same as first but explicit
                __DIR__ . '/../../../secure_keys.json',                 // In case of deeper nesting
                
                // System temp directory (fallback)
                sys_get_temp_dir() . '/securefacepay_keys.json',
                
                // Admin-specific location (if needed)
                dirname(__DIR__) . '/secure_keys.json'                  // admin_side/secure_keys.json
            ];
            
            foreach ($possiblePaths as $path) {
                $realPath = realpath($path);
                if ($realPath && file_exists($realPath)) {
                    self::$keyFile = $realPath;
                    error_log("AdminSecure: Found secure keys at: " . $realPath);
                    break;
                }
            }
            
            // Debug: Log all attempted paths if no keys found
            if (self::$keyFile === null) {
                error_log("AdminSecure: Could not find secure keys. Attempted paths:");
                foreach ($possiblePaths as $index => $path) {
                    $realPath = realpath($path) ?: $path;
                    $exists = file_exists($path) ? 'EXISTS' : 'NOT FOUND';
                    error_log("  Path {$index}: {$realPath} - {$exists}");
                }
                error_log("AdminSecure: Current __DIR__: " . __DIR__);
                error_log("AdminSecure: Current __FILE__: " . __FILE__);
            }
        }
        
        return self::$keyFile;
    }
    
    private static function loadKeys() {
        if (self::$keys !== null) {
            return self::$keys; // Already loaded
        }
        
        $keyFile = self::getKeyFilePath();
        
        // Check if keys file exists and is valid
        if ($keyFile && file_exists($keyFile)) {
            $keyData = file_get_contents($keyFile);
            $keys = json_decode($keyData, true);
            
            if ($keys && self::validateKeys($keys)) {
                self::$keys = $keys;
                error_log("AdminSecure: Loaded shared secure keys from " . $keyFile);
                return self::$keys;
            }
        }
        
        // If no secure keys found, admin must wait for customer migration
        error_log("ERROR: Admin cannot find secure keys. Please ensure customer_side migration is completed first.");
        throw new Exception("Secure keys not found. Please complete customer_side security migration first.");
    }
    
    private static function validateKeys($keys) {
        $requiredKeys = ['aes_key', 'aes_iv', 'password_key'];
        
        foreach ($requiredKeys as $key) {
            if (!isset($keys[$key]) || empty($keys[$key])) {
                return false;
            }
        }
        
        return true;
    }
    
    // Public methods to get specific keys (same interface as customer side)
    public static function getAESKey() {
        $keys = self::loadKeys();
        return base64_decode($keys['aes_key']);
    }
    
    public static function getAESIV() {
        $keys = self::loadKeys();
        return base64_decode($keys['aes_iv']);
    }
    
    public static function getPasswordKey() {
        $keys = self::loadKeys();
        return $keys['password_key'];
    }
    
    public static function getJWTSecret() {
        $keys = self::loadKeys();
        return $keys['jwt_secret'] ?? '';
    }
    
    // Admin-specific key info
    public static function getAdminKeyInfo() {
        return [
            'key_file_location' => self::getKeyFilePath(),
            'key_file_exists' => self::$keyFile ? file_exists(self::$keyFile) : false,
            'keys_loaded' => self::$keys !== null,
            'sharing_customer_keys' => true,
            'admin_security_status' => 'Secure - Using shared key infrastructure'
        ];
    }
}

// Database connection (enhanced with better error handling)
function dbConnect() {
    try {
        $conn = new mysqli('localhost', 'root', '', 'payment_facial');
        
        if ($conn->connect_error) {
            throw new Exception("Connection failed: " . $conn->connect_error);
        }
        
        $conn->set_charset("utf8mb4");
        return $conn;
        
    } catch (Exception $e) {
        error_log("Admin database connection failed: " . $e->getMessage());
        die("Database connection failed. Please try again later.");
    }
}

// SECURE Admin Password Deobfuscation Class - now using shared keys
class AdminPasswordDeobfuscator {
    private static function getKey() {
        return AdminSecureKeyManager::getPasswordKey();
    }
    
    public static function getPassword($field_name = 'password') {
        $obfuscated = $_POST[$field_name] ?? '';
        
        if (empty($obfuscated)) {
            return '';
        }
        
        // Check if it's obfuscated (starts with ENC_)
        if (!str_starts_with($obfuscated, 'ENC_')) {
            return $obfuscated; // Return as-is if not obfuscated
        }
        
        // Extract the obfuscated data
        $pattern = '/^ENC_[a-z0-9]+_(.+)_[a-z0-9]+_END$/';
        if (preg_match($pattern, $obfuscated, $matches)) {
            $encoded_data = $matches[1];
            
            try {
                // Decode and deobfuscate using shared secure key
                $decoded = base64_decode($encoded_data);
                $result = '';
                $key = self::getKey();
                
                for ($i = 0; $i < strlen($decoded); $i++) {
                    $charCode = ord($decoded[$i]);
                    $keyChar = ord($key[$i % strlen($key)]);
                    $deobfuscated = $charCode ^ $keyChar;
                    $result .= chr($deobfuscated);
                }
                
                return $result;
            } catch (Exception $e) {
                error_log("Admin password deobfuscation failed: " . $e->getMessage());
                return '';
            }
        }
        
        return ''; // Return empty if pattern doesn't match
    }
}

// Admin Rate Limiting (enhanced with better security)
function checkAdminRateLimit($identifier, $max_attempts, $time_window) {
    $conn = dbConnect();
    
    // Clean old attempts
    $stmt = $conn->prepare("DELETE FROM rate_limits WHERE identifier = ? AND attempt_time < ?");
    $cutoff_time = time() - $time_window;
    $stmt->bind_param("si", $identifier, $cutoff_time);
    $stmt->execute();
    
    // Count current attempts
    $stmt = $conn->prepare("SELECT COUNT(*) FROM rate_limits WHERE identifier = ?");
    $stmt->bind_param("s", $identifier);
    $stmt->execute();
    $result = $stmt->get_result();
    $count = $result->fetch_row()[0];
    
    if ($count >= $max_attempts) {
        $stmt->close();
        $conn->close();
        
        // Log admin rate limit exceeded
        logAdminSecurityEvent('system', 'RATE_LIMIT_EXCEEDED', 'BLOCKED', [
            'identifier' => $identifier,
            'attempts' => $count,
            'max_attempts' => $max_attempts
        ]);
        
        return false;
    }
    
    // Record this attempt
    $stmt = $conn->prepare("INSERT INTO rate_limits (identifier, attempt_time) VALUES (?, ?)");
    $current_time = time();
    $stmt->bind_param("si", $identifier, $current_time);
    $stmt->execute();
    
    $stmt->close();
    $conn->close();
    return true;
}

// Admin Security Event Logging (enhanced with admin-specific details)
function logAdminSecurityEvent($user_id, $action, $result, $details = []) {
    try {
        $conn = dbConnect();
        $stmt = $conn->prepare("
            INSERT INTO security_audit_log 
            (user_id, action, result, ip_address, user_agent, details, timestamp) 
            VALUES (?, ?, ?, ?, ?, ?, NOW())
        ");
        
        // Add admin context to details
        $admin_details = array_merge($details, [
            'admin_session' => $_SESSION['admin_session_token'] ?? 'none',
            'admin_role' => $_SESSION['admin_role'] ?? 'unknown',
            'admin_id' => $_SESSION['admin_id'] ?? 'unknown',
            'interface' => 'admin_panel'
        ]);
        
        $details_json = json_encode($admin_details);
        $ip = $_SERVER['REMOTE_ADDR'] ?? 'unknown';
        $user_agent = $_SERVER['HTTP_USER_AGENT'] ?? 'unknown';
        
        // Prefix admin events for easier filtering
        $admin_action = 'ADMIN_' . $action;
        
        $stmt->bind_param("ssssss", $user_id, $admin_action, $result, $ip, $user_agent, $details_json);
        $stmt->execute();
        $stmt->close();
        $conn->close();
    } catch (Exception $e) {
        error_log("Failed to log admin security event: " . $e->getMessage());
    }
}

// Admin CSRF Protection (enhanced)
function generateAdminCSRFToken() {
    if (!isset($_SESSION['admin_csrf_token'])) {
        $_SESSION['admin_csrf_token'] = bin2hex(random_bytes(32));
        $_SESSION['admin_csrf_token_time'] = time();
    }
    return $_SESSION['admin_csrf_token'];
}

function validateAdminCSRFToken($token) {
    if (!isset($_SESSION['admin_csrf_token']) || !isset($_SESSION['admin_csrf_token_time'])) {
        logAdminSecurityEvent($_SESSION['admin_id'] ?? 'unknown', 'CSRF_TOKEN_MISSING', 'FAILED');
        return false;
    }
    
    // 1 hour expiry
    if (time() - $_SESSION['admin_csrf_token_time'] > 3600) {
        unset($_SESSION['admin_csrf_token']);
        unset($_SESSION['admin_csrf_token_time']);
        logAdminSecurityEvent($_SESSION['admin_id'] ?? 'unknown', 'CSRF_TOKEN_EXPIRED', 'FAILED');
        return false;
    }
    
    $isValid = hash_equals($_SESSION['admin_csrf_token'], $token);
    
    if (!$isValid) {
        logAdminSecurityEvent($_SESSION['admin_id'] ?? 'unknown', 'CSRF_TOKEN_INVALID', 'FAILED', [
            'provided_token' => substr($token ?? '', 0, 8) . '...' // Log first 8 chars only
        ]);
    }
    
    return $isValid;
}

// SECURE encryption functions - now using shared secure keys
function aes_encrypt($plaintext) {
    return base64_encode(openssl_encrypt(
        $plaintext,
        'AES-256-CBC',
        AdminSecureKeyManager::getAESKey(),
        OPENSSL_RAW_DATA,
        AdminSecureKeyManager::getAESIV()
    ));
}

function aes_decrypt($ciphertext) {
    return openssl_decrypt(
        base64_decode($ciphertext),
        'AES-256-CBC',
        AdminSecureKeyManager::getAESKey(),
        OPENSSL_RAW_DATA,
        AdminSecureKeyManager::getAESIV()
    );
}

// Enhanced merchant name function with caching
function getMerchantName($merchant_id) {
    static $merchant_cache = [];
    
    if (isset($merchant_cache[$merchant_id])) {
        return $merchant_cache[$merchant_id];
    }
    
    $conn = dbConnect();
    $stmt = $conn->prepare("SELECT name FROM merchants WHERE merchant_id = ?");
    $stmt->bind_param("s", $merchant_id);
    $stmt->execute();
    $result = $stmt->get_result();
    $conn->close();
    
    $name = $result->num_rows > 0 ? $result->fetch_assoc()['name'] : 'Unknown Merchant';
    $merchant_cache[$merchant_id] = $name;
    
    return $name;
}

// Enhanced admin-specific configuration
define('ADMIN_SESSION_DURATION', 8 * 3600); // 8 hours
define('ADMIN_INACTIVITY_TIMEOUT', 2 * 3600); // 2 hours
define('ADMIN_MAX_LOGIN_ATTEMPTS', 3); // Stricter than customer side
define('ADMIN_LOCKOUT_TIME', 1800); // 30 minutes

// Enhanced role permissions with granular control
$ADMIN_PERMISSIONS = [
    'super_admin' => ['*'], // All permissions
    'financial_admin' => [
        'view_transactions', 
        'manage_wallets', 
        'generate_reports',
        'view_financial_analytics',
        'export_financial_data'
    ],
    'kiosk_admin' => [
        'manage_kiosks', 
        'manage_products', 
        'view_transactions',
        'manage_kiosk_settings',
        'view_kiosk_analytics'
    ],
    'support_admin' => [
        'view_users', 
        'manage_users', 
        'view_transactions',
        'handle_support_tickets',
        'view_user_analytics'
    ],
    'audit_admin' => [
        'view_audit_log', 
        'view_reports', 
        'view_transactions',
        'generate_audit_reports',
        'view_security_events'
    ],
    'read_only_admin' => [
        'view_dashboard',
        'view_reports',
        'view_transactions'
    ]
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
 * Enhanced admin session middleware with security logging
 */
function requireAdminAuth($required_roles = []) {
    $auth = getAdminAuth();
    
    // Check for session token
    $session_token = $_SESSION['admin_session_token'] ?? null;
    
    if (!$session_token) {
        logAdminSecurityEvent('anonymous', 'ACCESS_ATTEMPT_NO_TOKEN', 'BLOCKED');
        header("Location: login.php");
        exit();
    }
    
    // Validate session
    $admin = $auth->validateSession($session_token);
    if (!$admin) {
        logAdminSecurityEvent('unknown', 'INVALID_SESSION_TOKEN', 'BLOCKED', [
            'token_prefix' => substr($session_token, 0, 8) . '...'
        ]);
        header("Location: login.php");
        exit();
    }
    
    // Check role permissions
    if (!empty($required_roles) && $admin['role'] !== 'super_admin') {
        if (!in_array($admin['role'], $required_roles)) {
            logAdminSecurityEvent($admin['admin_id'], 'INSUFFICIENT_PERMISSIONS', 'BLOCKED', [
                'required_roles' => $required_roles,
                'user_role' => $admin['role']
            ]);
            http_response_code(403);
            die("Access denied: Insufficient permissions");
        }
    }
    
    // Log successful access
    logAdminSecurityEvent($admin['admin_id'] ?? 'unknown', 'PAGE_ACCESS', 'SUCCESS', [
        'page' => $_SERVER['REQUEST_URI'] ?? 'unknown',
        'role' => $admin['role'] ?? 'unknown'
    ]);
    
    return $admin;
}

/**
 * Check if admin has specific permission
 */
function hasAdminPermission($permission, $admin_role = null) {
    global $ADMIN_PERMISSIONS;
    
    if (!$admin_role) {
        $admin_role = $_SESSION['admin_role'] ?? null;
    }
    
    if (!$admin_role || !isset($ADMIN_PERMISSIONS[$admin_role])) {
        return false;
    }
    
    // Super admin has all permissions
    if ($admin_role === 'super_admin') {
        return true;
    }
    
    return in_array($permission, $ADMIN_PERMISSIONS[$admin_role]) || 
           in_array('*', $ADMIN_PERMISSIONS[$admin_role]);
}

// Enhanced session configuration for admin
session_name('FACEPAY_ADMIN_SESSION');

if (session_status() === PHP_SESSION_NONE) {
    session_set_cookie_params([
        'lifetime' => ADMIN_SESSION_DURATION,
        'path' => '/admin_side/', // Restrict to admin area only
        'secure' => isset($_SERVER['HTTPS']) && $_SERVER['HTTPS'] === 'on',
        'httponly' => true,
        'samesite' => 'Strict'
    ]);
    session_start();
    
    // Log session start
    if (!isset($_SESSION['admin_session_started'])) {
        $_SESSION['admin_session_started'] = time();
        logAdminSecurityEvent($_SESSION['admin_id'] ?? 'unknown', 'SESSION_STARTED', 'SUCCESS');
    }
}

// Admin security status check function
function getAdminSecurityStatus() {
    return [
        'key_manager' => AdminSecureKeyManager::getAdminKeyInfo(),
        'session_config' => [
            'session_name' => session_name(),
            'session_duration' => ADMIN_SESSION_DURATION,
            'inactivity_timeout' => ADMIN_INACTIVITY_TIMEOUT,
            'max_login_attempts' => ADMIN_MAX_LOGIN_ATTEMPTS
        ],
        'security_status' => 'Admin panel secured with shared key infrastructure',
        'timestamp' => date('Y-m-d H:i:s')
    ];
}

// Debug function for admin security status (localhost only)
function checkAdminSecurityStatus() {
    if (isset($_GET['admin_security_debug']) && $_GET['admin_security_debug'] === 'status' && 
        ($_SERVER['REMOTE_ADDR'] === '127.0.0.1' || $_SERVER['REMOTE_ADDR'] === '::1')) {
        
        header('Content-Type: application/json');
        echo json_encode([
            'status' => 'Admin Security Configuration Active',
            'admin_security' => getAdminSecurityStatus(),
            'message' => '🔐 Admin panel using shared secure infrastructure',
            'note' => 'Admin and customer systems share the same secure keys'
        ], JSON_PRETTY_PRINT);
        exit;
    }
}

// Check admin security status if requested
checkAdminSecurityStatus();

// Log successful admin config load
error_log("AdminSecure: Secure admin configuration loaded successfully - " . date('Y-m-d H:i:s'));

?>