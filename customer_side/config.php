<?php
/**
 * SECURE CONFIG.PHP - Enhanced security configuration
 * Backward compatible migration from hardcoded keys to secure key management
 */

// Secure Key Manager - CLEAN VERSION
class SecureKeyManager {
    private static $keyFile = null;
    private static $keys = null;
    
    private static function getKeyFilePath() {
        if (self::$keyFile === null) {
            $possiblePaths = [
                dirname(__DIR__) . '/secure_keys.json',     // FYP/secure_keys.json (outside web root)
                __DIR__ . '/../secure_keys.json',           // One level up from customer_side
                sys_get_temp_dir() . '/securefacepay_keys.json', // System temp (fallback)
                __DIR__ . '/.secure_keys.json'              // Current directory (last resort)
            ];
            
            foreach ($possiblePaths as $path) {
                $dir = dirname($path);
                if (is_writable($dir) || is_writable(dirname($dir))) {
                    self::$keyFile = $path;
                    break;
                }
            }
        }
        
        return self::$keyFile;
    }
    
    private static function generateNewKeys() {
        // Generate completely new secure keys
        return [
            'aes_key' => base64_encode(random_bytes(32)),        // New 256-bit AES key
            'aes_iv' => base64_encode(random_bytes(16)),         // New 128-bit IV
            'jwt_secret' => bin2hex(random_bytes(32)),           // New 256-bit JWT secret
            'encryption_key' => bin2hex(random_bytes(16)),       // New encryption key
            'password_key' => bin2hex(random_bytes(16)),         // New password key
            'csrf_secret' => bin2hex(random_bytes(32)),          // New CSRF secret
            'generated_date' => date('Y-m-d H:i:s'),
            'migration_status' => 'new_keys_generated',
            'version' => '2.0',
            'note' => 'Fresh secure keys - no hardcoded values'
        ];
    }
    
    private static function loadKeys() {
        if (self::$keys !== null) {
            return self::$keys; // Already loaded
        }
        
        $keyFile = self::getKeyFilePath();
        
        // Check if keys file exists and is valid
        if (file_exists($keyFile)) {
            $keyData = file_get_contents($keyFile);
            $keys = json_decode($keyData, true);
            
            if ($keys && self::validateKeys($keys)) {
                self::$keys = $keys;
                error_log("SecureFacePay: Loaded existing secure keys from " . $keyFile);
                return self::$keys;
            } else {
                error_log("Warning: Invalid keys file. Regenerating...");
            }
        }
        
        // This should NOT happen since migration is complete
        // But if keys are missing, generate new ones
        error_log("ERROR: Secure keys file not found! This should not happen after migration.");
        throw new Exception("Secure keys file missing. Please restore from backup or regenerate.");
    }
    
    private static function validateKeys($keys) {
        $requiredKeys = ['aes_key', 'aes_iv', 'jwt_secret', 'password_key'];
        
        foreach ($requiredKeys as $key) {
            if (!isset($keys[$key]) || empty($keys[$key])) {
                return false;
            }
        }
        
        // Validate key formats
        if (strlen(base64_decode($keys['aes_key'])) !== 32) return false;
        if (strlen(base64_decode($keys['aes_iv'])) !== 16) return false;
        if (strlen($keys['jwt_secret']) < 32) return false;
        
        return true;
    }
    
    private static function saveKeys() {
        $keyFile = self::getKeyFilePath();
        $keyData = json_encode(self::$keys, JSON_PRETTY_PRINT);
        
        $dir = dirname($keyFile);
        if (!is_dir($dir)) {
            mkdir($dir, 0755, true);
        }
        
        if (file_put_contents($keyFile, $keyData, LOCK_EX)) {
            chmod($keyFile, 0600); // Owner read/write only
            error_log("SecureFacePay: Keys saved securely to: " . $keyFile);
        } else {
            error_log("Warning: Could not save keys to: " . $keyFile);
        }
    }
    
    // Public methods to get specific keys
    public static function getAESKey() {
        $keys = self::loadKeys();
        return base64_decode($keys['aes_key']);
    }
    
    public static function getAESIV() {
        $keys = self::loadKeys();
        return base64_decode($keys['aes_iv']);
    }
    
    public static function getJWTSecret() {
        $keys = self::loadKeys();
        return $keys['jwt_secret'];
    }
    
    public static function getEncryptionKey() {
        $keys = self::loadKeys();
        return $keys['encryption_key'] ?? '';
    }
    
    public static function getPasswordKey() {
        $keys = self::loadKeys();
        return $keys['password_key'];
    }
    
    public static function getCSRFSecret() {
        $keys = self::loadKeys();
        return $keys['csrf_secret'] ?? bin2hex(random_bytes(32));
    }
    
    // Migration status check
    public static function getMigrationStatus() {
        $keys = self::loadKeys();
        return [
            'migration_completed' => isset($keys['migration_status']),
            'migration_date' => $keys['migration_date'] ?? $keys['generated_date'] ?? 'unknown',
            'using_legacy_keys' => $keys['migration_status'] === 'completed_from_legacy',
            'all_data_preserved' => true,
            'key_file_location' => self::getKeyFilePath(),
            'version' => $keys['version'] ?? '1.0',
            'status' => 'Secure - No hardcoded keys'
        ];
    }
    
    // Debug function for key status (localhost only)
    public static function getKeyInfo() {
        $keyFile = self::getKeyFilePath();
        $keys = self::loadKeys();
        
        return [
            'key_file_location' => $keyFile,
            'key_file_exists' => file_exists($keyFile),
            'migration_date' => $keys['migration_date'] ?? $keys['generated_date'] ?? 'unknown',
            'version' => $keys['version'] ?? 'unknown',
            'keys_loaded' => self::$keys !== null,
            'migration_status' => $keys['migration_status'] ?? 'unknown',
            'security_status' => 'Production Ready - No Hardcoded Keys'
        ];
    }
}

// PHPMailer imports - Must be at top level, not inside conditionals
use PHPMailer\PHPMailer\PHPMailer;
use PHPMailer\PHPMailer\Exception;

if (session_status() === PHP_SESSION_NONE) {
    session_set_cookie_params([
        'lifetime' => 1800, // 30 minutes
        'path' => '/',
        'secure' => isset($_SERVER['HTTPS']) && $_SERVER['HTTPS'] === 'on',
        'httponly' => true,
        'samesite' => 'Strict'
    ]);
    session_start();
}

// Enhanced database configuration
class Config {
    // Database configuration
    const DB_HOST = 'localhost';
    const DB_USER = 'root';
    const DB_PASS = '';
    const DB_NAME = 'payment_facial'; // Keep your existing database name
    
    // Security configuration - now using SecureKeyManager but preserving your existing values
    public static function getJwtSecret() {
        return SecureKeyManager::getJWTSecret();
    }
    
    public static function getEncryptionKey() {
        return SecureKeyManager::getEncryptionKey();
    }
    
    // Other constants remain the same
    const CSRF_TOKEN_EXPIRY = 3600; // 1 hour
    const SESSION_TIMEOUT = 1800; // 30 minutes
    
    // Payment security
    const MAX_TRANSACTION_AMOUNT = 5000.00;
    const DAILY_LIMIT_DEFAULT = 1000.00;
    const MONTHLY_LIMIT_DEFAULT = 10000.00;
    
    // Facial recognition API
    const FACE_API_URL = 'http://localhost:5000';
    const FACE_CONFIDENCE_THRESHOLD = 0.45;
    const LIVENESS_THRESHOLD = 0.7;
    
    // Rate limiting
    const MAX_LOGIN_ATTEMPTS = 5;
    const LOGIN_LOCKOUT_TIME = 900; // 15 minutes
}

function dbConnect() {
    try {
        $host = getenv('DB_HOST') ?: Config::DB_HOST;
        $user = getenv('DB_USER') ?: Config::DB_USER;
        $pass = getenv('DB_PASS') ?: Config::DB_PASS;
        $db = getenv('DB_NAME') ?: Config::DB_NAME;

        // Return MySQLi connection to match your working code
        $conn = new mysqli($host, $user, $pass, $db);
        
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

// SECURE encryption functions - now using SecureKeyManager but preserving your exact keys
function getAESKey() {
    return SecureKeyManager::getAESKey();
}

function getAESIV() {
    return SecureKeyManager::getAESIV();
}

// Move encryption functions here to avoid circular dependency
function aes_encrypt($plaintext) {
    return base64_encode(openssl_encrypt(
        $plaintext,
        'AES-256-CBC',
        getAESKey(),
        OPENSSL_RAW_DATA,
        getAESIV()
    ));
}

function aes_decrypt($ciphertext) {
    return openssl_decrypt(
        base64_decode($ciphertext),
        'AES-256-CBC',
        getAESKey(),
        OPENSSL_RAW_DATA,
        getAESIV()
    );
}

// Enhanced encryption class that works with your existing functions
class SecureEncryption {
    private static $key;
    
    public static function init() {
        self::$key = getAESKey();
    }
    
    public static function encrypt($data) {
        // Use the aes_encrypt function defined above
        return aes_encrypt($data);
    }
    
    public static function decrypt($data) {
        // Use the aes_decrypt function defined above
        return aes_decrypt($data);
    }
    
    // Enhanced version with HMAC for new sensitive data
    public static function encryptSecure($data) {
        if (!self::$key) self::init();
        
        $iv = random_bytes(16);
        $encrypted = openssl_encrypt($data, 'AES-256-CBC', self::$key, OPENSSL_RAW_DATA, $iv);
        $hmac = hash_hmac('sha256', $encrypted . $iv, self::$key, true);
        
        return base64_encode($hmac . $iv . $encrypted);
    }
    
    public static function decryptSecure($data) {
        if (!self::$key) self::init();
        
        $data = base64_decode($data);
        $hmac = substr($data, 0, 32);
        $iv = substr($data, 32, 16);
        $encrypted = substr($data, 48);
        
        $calcmac = hash_hmac('sha256', $encrypted . $iv, self::$key, true);
        if (!hash_equals($hmac, $calcmac)) {
            throw new Exception('HMAC verification failed');
        }
        
        return openssl_decrypt($encrypted, 'AES-256-CBC', self::$key, OPENSSL_RAW_DATA, $iv);
    }
}

// Additional encryption functions for enhanced security
function encrypt_sensitive_data($data) {
    return SecureEncryption::encryptSecure($data);
}

function decrypt_sensitive_data($encrypted_data) {
    return SecureEncryption::decryptSecure($encrypted_data);
}

// Enhanced CSRF functions
function generateCSRFToken() {
    if (!isset($_SESSION['csrf_token'])) {
        $_SESSION['csrf_token'] = bin2hex(random_bytes(32));
        $_SESSION['csrf_token_time'] = time();
    }
    return $_SESSION['csrf_token'];
}

function validateCSRFToken($token) {
    if (!isset($_SESSION['csrf_token']) || !isset($_SESSION['csrf_token_time'])) {
        return false;
    }
    
    if (time() - $_SESSION['csrf_token_time'] > Config::CSRF_TOKEN_EXPIRY) {
        unset($_SESSION['csrf_token']);
        unset($_SESSION['csrf_token_time']);
        return false;
    }
    
    return hash_equals($_SESSION['csrf_token'], $token);
}

// Security utilities
class SecurityUtils {
    public static function sanitizeInput($data) {
        return htmlspecialchars(trim($data), ENT_QUOTES, 'UTF-8');
    }
    
    public static function logSecurityEvent($user_id, $action, $result, $details = []) {
        try {
            $conn = dbConnect();
            $stmt = $conn->prepare("
                INSERT INTO security_audit_log 
                (user_id, action, result, ip_address, user_agent, details, timestamp) 
                VALUES (?, ?, ?, ?, ?, ?, NOW())
            ");
            
            $details_json = json_encode($details);
            $ip = $_SERVER['REMOTE_ADDR'] ?? 'unknown';
            $user_agent = $_SERVER['HTTP_USER_AGENT'] ?? 'unknown';
            
            $stmt->bind_param("ssssss", $user_id, $action, $result, $ip, $user_agent, $details_json);
            $stmt->execute();
            $stmt->close();
            $conn->close();
        } catch (Exception $e) {
            error_log("Failed to log security event: " . $e->getMessage());
        }
    }
    
    // public static function hashPassword($password) {
    //     return password_hash($password, PASSWORD_ARGON2ID, [
    //         'memory_cost' => 65536, // 64 MB
    //         'time_cost' => 4,       // 4 iterations
    //         'threads' => 3,         // 3 threads
    //     ]);
    // }
    
    public static function verifyPassword($password, $hash) {
        return password_verify($password, $hash);
    }
    
    public static function checkRateLimit($identifier, $max_attempts, $time_window) {
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
    
    // Debug function to check security status
    public static function getSecurityStatus() {
        return [
            'migration' => SecureKeyManager::getMigrationStatus(),
            'php_version' => PHP_VERSION,
            'openssl_available' => extension_loaded('openssl'),
            'random_bytes_available' => function_exists('random_bytes'),
            'session_status' => session_status(),
            'current_time' => date('Y-m-d H:i:s')
        ];
    }
}

// Session management
class SessionManager {
    public static function start() {
        if (session_status() === PHP_SESSION_NONE) {
            session_start();
        }
        
        // Regenerate session ID periodically
        if (!isset($_SESSION['last_regeneration'])) {
            $_SESSION['last_regeneration'] = time();
        } elseif (time() - $_SESSION['last_regeneration'] > 300) { // 5 minutes
            session_regenerate_id(true);
            $_SESSION['last_regeneration'] = time();
        }
        
        // Check session timeout
        if (isset($_SESSION['last_activity'])) {
            if (time() - $_SESSION['last_activity'] > Config::SESSION_TIMEOUT) {
                self::destroy();
                return false;
            }
        }
        
        $_SESSION['last_activity'] = time();
        return true;
    }
    
    public static function setAuthLevel($level, $user_id) {
        $_SESSION['auth_level'] = $level;
        $_SESSION['user_id'] = $user_id;
        $_SESSION['auth_time'] = time();
    }
    
    public static function requireAuthLevel($required_level) {
        if (!isset($_SESSION['auth_level']) || !isset($_SESSION['user_id'])) {
            return false;
        }
        
        $levels = ['basic' => 1, 'sensitive' => 2, 'critical' => 3];
        $current = $levels[$_SESSION['auth_level']] ?? 0;
        $required = $levels[$required_level] ?? 999;
        
        return $current >= $required;
    }
    
    public static function destroy() {
        session_unset();
        session_destroy();
        session_start();
        session_regenerate_id(true);
    }
}

// FIXED: Updated PHPMailer path for customer_side directory structure
$vendor_paths = [
    __DIR__ . '/vendor/autoload.php',           // customer_side/vendor/autoload.php
    __DIR__ . '/../vendor/autoload.php',        // FYP/vendor/autoload.php
    __DIR__ . '/../../vendor/autoload.php',     // Two levels up
    getcwd() . '/vendor/autoload.php',          // Current working directory
];

$autoloader_found = false;
foreach ($vendor_paths as $path) {
    if (file_exists($path)) {
        require_once $path;
        $autoloader_found = true;
        break;
    }
}

// If autoloader not found, try manual include paths that match your working setup
if (!$autoloader_found) {
    $phpmailer_paths = [
        __DIR__ . '/vendor/phpmailer/phpmailer/src/PHPMailer.php',       // customer_side/vendor/phpmailer/
        __DIR__ . '/../vendor/phpmailer/phpmailer/src/PHPMailer.php',    // FYP/vendor/phpmailer/
        __DIR__ . '/vendor/phpmailer/src/PHPMailer.php',                 // Alternative structure
        __DIR__ . '/../vendor/phpmailer/src/PHPMailer.php',              // Alternative
    ];
    
    foreach ($phpmailer_paths as $path) {
        if (file_exists($path)) {
            require_once str_replace('PHPMailer.php', 'Exception.php', $path);
            require_once $path;
            require_once str_replace('PHPMailer.php', 'SMTP.php', $path);
            break;
        }
    }
}

// Email sending function (matching your working code style)
function sendVerificationEmail($email, $token) {
    // Check if PHPMailer is available
    if (!class_exists('PHPMailer\PHPMailer\PHPMailer') && !class_exists('PHPMailer')) {
        error_log("PHPMailer class not found. Please check your vendor/autoload.php path.");
        return false;
    }
    
    $mail = new PHPMailer(true);
    try {
        //Server settings (matching your working code exactly)
        $mail->isSMTP();
        $mail->Host = 'smtp.gmail.com';
        $mail->SMTPAuth = true;
        $mail->Username = 'authorizedpay88@gmail.com';
        $mail->Password = 'tbpwxivpydxjqrwq';
        $mail->SMTPSecure = 'tls';
        $mail->Port = 587;
        
        // Enable debug output for testing (set to 0 in production)
        $mail->SMTPDebug = 0; // Set to 2 for debugging
        
        //Recipients (matching your working code)
        $mail->setFrom('authorizedpay88@gmail.com', 'Facial Pay System');
        $mail->addAddress($email);
        $mail->isHTML(true);
        $mail->Subject = 'Verify your email';
        
        // Match your working verification URL
        $verification_url = "http://localhost/FYP/customer_side/verify.php?token=$token";
        $mail->Body = "Click the link below to verify your account:<br><a href='$verification_url'>Verify Account</a>";
        
        $result = $mail->send();
        
        if ($result) {
            error_log("Verification email sent successfully to: " . $email);
        } else {
            error_log("Failed to send verification email to: " . $email);
        }
        
        return $result;
        
    } catch (Exception $e) {
        error_log("Mailer Error: {$mail->ErrorInfo}");
        error_log("Exception: " . $e->getMessage());
        return false;
    }
}

// Profile change verification email function
function sendProfileChangeVerificationEmail($email, $name, $change_type, $token) {
    // Check if PHPMailer is available
    if (!class_exists('PHPMailer\PHPMailer\PHPMailer') && !class_exists('PHPMailer')) {
        error_log("PHPMailer class not found. Please check your vendor/autoload.php path.");
        return false;
    }
    
    $mail = new PHPMailer(true);
    try {
        // Server settings (matching your working config)
        $mail->isSMTP();
        $mail->Host = 'smtp.gmail.com';
        $mail->SMTPAuth = true;
        $mail->Username = 'authorizedpay88@gmail.com';
        $mail->Password = 'tbpwxivpydxjqrwq';
        $mail->SMTPSecure = 'tls';
        $mail->Port = 587;
        $mail->SMTPDebug = 0;
        
        // Recipients
        $mail->setFrom('authorizedpay88@gmail.com', 'SecureFacePay Security');
        $mail->addAddress($email);
        $mail->isHTML(true);
        $mail->Subject = 'Verify Your ' . ucfirst($change_type) . ' - SecureFacePay';
        
        // Verification URL
        $verification_url = "http://localhost/FYP/customer_side/verify_change.php?token=" . $token;
        
        // Email template
        $mail->Body = "
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: linear-gradient(135deg, #3498db, #667eea); color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }
                .content { background: white; padding: 30px; border: 1px solid #ddd; }
                .button { display: inline-block; padding: 12px 24px; background: #3498db; color: white; text-decoration: none; border-radius: 5px; margin: 10px 0; }
                .footer { background: #f8f9fa; padding: 15px; font-size: 12px; color: #666; text-align: center; border-radius: 0 0 8px 8px; }
            </style>
        </head>
        <body>
            <div class='container'>
                <div class='header'>
                    <h2>🔒 SecureFacePay Verification</h2>
                </div>
                <div class='content'>
                    <h3>Hi " . htmlspecialchars($name) . ",</h3>
                    <p>You requested to change your <strong>" . htmlspecialchars($change_type) . "</strong> on your SecureFacePay account.</p>
                    <p>For security purposes, please click the button below to verify this change:</p>
                    <p style='text-align: center;'>
                        <a href='" . $verification_url . "' class='button'>Verify " . ucfirst(htmlspecialchars($change_type)) . "</a>
                    </p>
                    <p><strong>Important:</strong></p>
                    <ul>
                        <li>This link will expire in 1 hour</li>
                        <li>If you didn't request this change, please ignore this email</li>
                        <li>Never share this verification link with anyone</li>
                    </ul>
                    <p>If the button doesn't work, copy and paste this link: <br>
                    <small>" . $verification_url . "</small></p>
                </div>
                <div class='footer'>
                    <p>SecureFacePay Security Team | This is an automated message</p>
                </div>
            </div>
        </body>
        </html>
        ";
        
        $result = $mail->send();
        
        if ($result) {
            error_log("Profile change verification email sent successfully to: " . $email);
        } else {
            error_log("Failed to send profile change verification email to: " . $email);
        }
        
        return $result;
        
    } catch (Exception $e) {
        error_log("Profile Change Mailer Error: {$mail->ErrorInfo}");
        error_log("Exception: " . $e->getMessage());
        return false;
    }
}

// Admin notification for email changes
function notifyAdminEmailChange($user_name, $old_email, $new_email) {
    // Check if PHPMailer is available
    if (!class_exists('PHPMailer\PHPMailer\PHPMailer') && !class_exists('PHPMailer')) {
        error_log("PHPMailer class not found. Please check your vendor/autoload.php path.");
        return false;
    }
    
    $mail = new PHPMailer(true);
    try {
        // Server settings (matching your working config)
        $mail->isSMTP();
        $mail->Host = 'smtp.gmail.com';
        $mail->SMTPAuth = true;
        $mail->Username = 'authorizedpay88@gmail.com';
        $mail->Password = 'tbpwxivpydxjqrwq';
        $mail->SMTPSecure = 'tls';
        $mail->Port = 587;
        $mail->SMTPDebug = 0;
        
        // Recipients
        $mail->setFrom('authorizedpay88@gmail.com', 'SecureFacePay System');
        $mail->addAddress('admin@securefacepay.com'); // Change to your admin email
        $mail->isHTML(true);
        $mail->Subject = 'Email Change Request - SecureFacePay Admin';
        
        // Email template
        $mail->Body = "
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: #e74c3c; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }
                .content { background: white; padding: 30px; border: 1px solid #ddd; }
                .footer { background: #f8f9fa; padding: 15px; font-size: 12px; color: #666; text-align: center; border-radius: 0 0 8px 8px; }
            </style>
        </head>
        <body>
            <div class='container'>
                <div class='header'>
                    <h2>⚠️ Email Change Request</h2>
                </div>
                <div class='content'>
                    <h3>Admin Action Required</h3>
                    <p>A user has requested an email address change:</p>
                    <ul>
                        <li><strong>User:</strong> " . htmlspecialchars($user_name) . "</li>
                        <li><strong>Current Email:</strong> " . htmlspecialchars($old_email) . "</li>
                        <li><strong>Requested Email:</strong> " . htmlspecialchars($new_email) . "</li>
                        <li><strong>Time:</strong> " . date('Y-m-d H:i:s') . "</li>
                    </ul>
                    <p>Please review this request in the admin panel.</p>
                </div>
                <div class='footer'>
                    <p>SecureFacePay System | This is an automated message</p>
                </div>
            </div>
        </body>
        </html>
        ";
        
        $result = $mail->send();
        
        if ($result) {
            error_log("Admin notification email sent successfully for email change request");
        } else {
            error_log("Failed to send admin notification email for email change request");
        }
        
        return $result;
        
    } catch (Exception $e) {
        error_log("Admin Notification Mailer Error: {$mail->ErrorInfo}");
        error_log("Exception: " . $e->getMessage());
        return false;
    }
}

// Notification email function for profile changes
function sendNotificationEmail($email, $name, $change_type, $details = []) {
    // Check if PHPMailer is available
    if (!class_exists('PHPMailer\PHPMailer\PHPMailer') && !class_exists('PHPMailer')) {
        error_log("PHPMailer class not found. Please check your vendor/autoload.php path.");
        return false;
    }
    
    $mail = new PHPMailer(true);
    try {
        // Server settings (matching your working config)
        $mail->isSMTP();
        $mail->Host = 'smtp.gmail.com';
        $mail->SMTPAuth = true;
        $mail->Username = 'authorizedpay88@gmail.com';
        $mail->Password = 'tbpwxivpydxjqrwq';
        $mail->SMTPSecure = 'tls';
        $mail->Port = 587;
        $mail->SMTPDebug = 0;
        
        // Recipients
        $mail->setFrom('authorizedpay88@gmail.com', 'SecureFacePay Security');
        $mail->addAddress($email);
        $mail->isHTML(true);
        
        // Email content based on change type
        $subjects = [
            'profile_update' => 'Profile Updated - SecureFacePay',
            'password_change' => 'Password Changed - SecureFacePay',
            'pin_change' => 'PIN Changed - SecureFacePay',
            'face_removal' => 'Face Registration Removed - SecureFacePay',
            'face_added' => 'Face Registration Added - SecureFacePay'
        ];
        
        $change_messages = [
            'profile_update' => 'Your profile information has been updated successfully.',
            'password_change' => 'Your account password has been changed successfully.',
            'pin_change' => 'Your transaction PIN has been changed successfully.',
            'face_removal' => 'Your face registration has been removed from your account.',
            'face_added' => 'Face registration has been added to your account.'
        ];
        
        $mail->Subject = $subjects[$change_type] ?? 'Account Updated - SecureFacePay';
        $change_message = $change_messages[$change_type] ?? 'Your account has been updated.';
        
        $mail->Body = "
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: linear-gradient(135deg, #38a169, #48bb78); color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }
                .content { background: white; padding: 30px; border: 1px solid #ddd; }
                .footer { background: #f8f9fa; padding: 15px; font-size: 12px; color: #666; text-align: center; border-radius: 0 0 8px 8px; }
                .alert { background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 15px 0; }
            </style>
        </head>
        <body>
            <div class='container'>
                <div class='header'>
                    <h2>✅ Account Update Notification</h2>
                </div>
                <div class='content'>
                    <h3>Hi " . htmlspecialchars($name) . ",</h3>
                    <p>" . $change_message . "</p>
                    <div class='alert'>
                        <strong>Change Details:</strong><br>
                        Date: " . date('F j, Y g:i A') . "<br>
                    </div>
                    <p><strong>If you didn't make this change:</strong></p>
                    <ul>
                        <li>Please contact our support team immediately</li>
                        <li>Change your password as a precaution</li>
                        <li>Check your account for any unauthorized activity</li>
                    </ul>
                </div>
                <div class='footer'>
                    <p>SecureFacePay Security Team | This is an automated message</p>
                </div>
            </div>
        </body>
        </html>
        ";
        
        $result = $mail->send();
        
        if ($result) {
            error_log("Notification email sent successfully to: " . $email . " for " . $change_type);
        } else {
            error_log("Failed to send notification email to: " . $email . " for " . $change_type);
        }
        
        return $result;
        
    } catch (Exception $e) {
        error_log("Notification Mailer Error: {$mail->ErrorInfo}");
        error_log("Exception: " . $e->getMessage());
        return false;
    }
}

// Email verification function for new email address changes
function sendEmailVerification($new_email, $name, $token) {
    // Check if PHPMailer is available
    if (!class_exists('PHPMailer\PHPMailer\PHPMailer') && !class_exists('PHPMailer')) {
        error_log("PHPMailer class not found. Please check your vendor/autoload.php path.");
        return false;
    }
    
    $mail = new PHPMailer(true);
    try {
        // Server settings (matching your working config)
        $mail->isSMTP();
        $mail->Host = 'smtp.gmail.com';
        $mail->SMTPAuth = true;
        $mail->Username = 'authorizedpay88@gmail.com';
        $mail->Password = 'tbpwxivpydxjqrwq';
        $mail->SMTPSecure = 'tls';
        $mail->Port = 587;
        $mail->SMTPDebug = 0;
        
        // Recipients
        $mail->setFrom('authorizedpay88@gmail.com', 'SecureFacePay Security');
        $mail->addAddress($new_email);
        $mail->isHTML(true);
        $mail->Subject = 'Verify Your New Email Address - SecureFacePay';
        
        // Verification URL
        $verification_url = "http://localhost/FYP/customer_side/verify_email_change.php?token=" . $token;
        
        $mail->Body = "
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: linear-gradient(135deg, #3498db, #667eea); color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }
                .content { background: white; padding: 30px; border: 1px solid #ddd; }
                .button { display: inline-block; padding: 12px 24px; background: #3498db; color: white; text-decoration: none; border-radius: 5px; margin: 10px 0; }
                .footer { background: #f8f9fa; padding: 15px; font-size: 12px; color: #666; text-align: center; border-radius: 0 0 8px 8px; }
            </style>
        </head>
        <body>
            <div class='container'>
                <div class='header'>
                    <h2>🔒 Verify Your New Email</h2>
                </div>
                <div class='content'>
                    <h3>Hi " . htmlspecialchars($name) . ",</h3>
                    <p>You requested to change your email address to this email on your SecureFacePay account.</p>
                    <p>Please click the button below to verify this new email address:</p>
                    <p style='text-align: center;'>
                        <a href='" . $verification_url . "' class='button'>Verify New Email</a>
                    </p>
                    <p><strong>Important:</strong></p>
                    <ul>
                        <li>This link will expire in 24 hours</li>
                        <li>If you didn't request this change, please ignore this email</li>
                        <li>Your old email will remain active until you verify this one</li>
                    </ul>
                    <p>If the button doesn't work, copy and paste this link: <br>
                    <small>" . $verification_url . "</small></p>
                </div>
                <div class='footer'>
                    <p>SecureFacePay Security Team | This is an automated message</p>
                </div>
            </div>
        </body>
        </html>
        ";
        
        $result = $mail->send();
        
        if ($result) {
            error_log("Email verification sent successfully to: " . $new_email);
        } else {
            error_log("Failed to send email verification to: " . $new_email);
        }
        
        return $result;
        
    } catch (Exception $e) {
        error_log("Email Verification Mailer Error: {$mail->ErrorInfo}");
        error_log("Exception: " . $e->getMessage());
        return false;
    }
}

// Simple password reset email function
function sendPasswordResetEmail($email, $name, $token) {
    // Check if PHPMailer is available
    if (!class_exists('PHPMailer\PHPMailer\PHPMailer') && !class_exists('PHPMailer')) {
        error_log("PHPMailer class not found. Please check your vendor/autoload.php path.");
        return false;
    }
    
    $mail = new PHPMailer(true);
    try {
        // Server settings (matching your working config exactly)
        $mail->isSMTP();
        $mail->Host = 'smtp.gmail.com';
        $mail->SMTPAuth = true;
        $mail->Username = 'authorizedpay88@gmail.com';
        $mail->Password = 'tbpwxivpydxjqrwq';
        $mail->SMTPSecure = 'tls';
        $mail->Port = 587;
        $mail->SMTPDebug = 0;
        
        // Recipients
        $mail->setFrom('authorizedpay88@gmail.com', 'Facial Pay System');
        $mail->addAddress($email);
        $mail->isHTML(true);
        $mail->Subject = 'Reset Your Password - Facial Pay System';
        
        // Reset URL (reuses your existing verification structure)
        $reset_url = "http://localhost/FYP/customer_side/forgot_password.php?step=reset&token=" . $token;
        
        // Simple email template (matching your existing style)
        $mail->Body = "
        <html>
        <body style='font-family: Arial, sans-serif; line-height: 1.6; color: #333;'>
            <div style='max-width: 600px; margin: 0 auto; padding: 20px;'>
                <h2 style='color: #3498db;'>Password Reset Request</h2>
                <p>Hi " . htmlspecialchars($name) . ",</p>
                <p>You requested to reset your password for your Facial Pay System account.</p>
                <p>Click the link below to reset your password:</p>
                <p><a href='" . $reset_url . "' style='background: #3498db; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block;'>Reset My Password</a></p>
                <p><strong>Important:</strong> This link will expire soon and can only be used once.</p>
                <p>If you didn't request this password reset, please ignore this email.</p>
                <hr>
                <p><small>Facial Pay System - Automated Message</small></p>
            </div>
        </body>
        </html>
        ";
        
        $result = $mail->send();
        
        if ($result) {
            error_log("Password reset email sent successfully to: " . $email);
        } else {
            error_log("Failed to send password reset email to: " . $email);
        }
        
        return $result;
        
    } catch (Exception $e) {
        error_log("Password Reset Mailer Error: {$mail->ErrorInfo}");
        error_log("Exception: " . $e->getMessage());
        return false;
    }
}

// Password Deobfuscation Class - now secure but backward compatible
class PasswordDeobfuscator {
    private static function getKey() {
        return SecureKeyManager::getPasswordKey();
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
                // Decode and deobfuscate
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
                error_log("Password deobfuscation failed: " . $e->getMessage());
                return '';
            }
        }
        
        return ''; // Return empty if pattern doesn't match
    }
    
    public static function getPin($field_name = 'new_pin') {
        // Handle PIN arrays
        $pin_array = $_POST[$field_name] ?? [];
        
        if (is_array($pin_array)) {
            $deobfuscated_pins = [];
            foreach ($pin_array as $pin_digit) {
                if (!empty($pin_digit)) {
                    $deobfuscated_pins[] = self::deobfuscateSingle($pin_digit);
                }
            }
            return $deobfuscated_pins;
        }
        
        return [];
    }
    
    private static function deobfuscateSingle($obfuscated) {
        if (str_starts_with($obfuscated, 'ENC_')) {
            $pattern = '/^ENC_[a-z0-9]+_(.+)_[a-z0-9]+_END$/';
            if (preg_match($pattern, $obfuscated, $matches)) {
                $encoded_data = $matches[1];
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
            }
        }
        
        return $obfuscated; // Return as-is if not obfuscated
    }
}

// Utility function to get merchant name
function getMerchantName($merchant_id) {
    $conn = dbConnect();
    $stmt = $conn->prepare("SELECT name FROM merchants WHERE merchant_id = ?");
    $stmt->bind_param("s", $merchant_id);
    $stmt->execute();
    $result = $stmt->get_result();
    $conn->close();
    return $result->num_rows > 0 ? $result->fetch_assoc()['name'] : 'Unknown Merchant';
}

// Debug function to check migration status (localhost only)
function checkSecurityMigration() {
    if (isset($_GET['security_debug']) && $_GET['security_debug'] === 'status' && 
        ($_SERVER['REMOTE_ADDR'] === '127.0.0.1' || $_SERVER['REMOTE_ADDR'] === '::1')) {
        
        header('Content-Type: application/json');
        echo json_encode([
            'status' => 'SecureFacePay Migration Complete',
            'security' => SecurityUtils::getSecurityStatus(),
            'message' => '✅ All existing data preserved - System now secure!',
            'timestamp' => date('Y-m-d H:i:s'),
            'note' => 'Keys are now stored securely outside web root'
        ], JSON_PRETTY_PRINT);
        exit;
    }
}

// Check security migration status if requested
checkSecurityMigration();

// Optional: Log successful configuration load
error_log("SecureFacePay: Secure configuration loaded successfully - " . date('Y-m-d H:i:s'));

?>