<?php
// Updated config.php - Enhanced security configuration

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
    
    // Security configuration
    const JWT_SECRET = '9b98936e7efd1fec5ede6997b6fc3b34e020c429555574bd235174130ec24fd5';
    const ENCRYPTION_KEY = '5d8755cef84e1c81991688ff4e88d00a';
    const CSRF_TOKEN_EXPIRY = 3600; // 1 hour
    const SESSION_TIMEOUT = 1800; // 30 minutes
    
    // Payment security
    const MAX_TRANSACTION_AMOUNT = 5000.00;
    const DAILY_LIMIT_DEFAULT = 1000.00;
    const MONTHLY_LIMIT_DEFAULT = 10000.00;
    
    // Facial recognition API
    const FACE_API_URL = 'http://localhost:5000';
    const FACE_CONFIDENCE_THRESHOLD = 0.7;
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

// Keep your existing AES functions but enhance them
function getAESKey() {
    return base64_decode(getenv('AES_KEY') ?: '0KvocVvCxtHUCXBQ8LVuztXHBFy7fiyjzhxOsXxQCmU=');
}

function getAESIV() {
    return base64_decode(getenv('AES_IV') ?: '0OUi5xYbQkZgId+yqOrCUQ==');
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
    
    public static function hashPassword($password) {
        return password_hash($password, PASSWORD_ARGON2ID, [
            'memory_cost' => 65536, // 64 MB
            'time_cost' => 4,       // 4 iterations
            'threads' => 3,         // 3 threads
        ]);
    }
    
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
// Your working code uses 'vendor/autoload.php' which means vendor should be in customer_side
$vendor_paths = [
    __DIR__ . '/vendor/autoload.php',           // customer_side/vendor/autoload.php (if you moved vendor here)
    __DIR__ . '/../vendor/autoload.php',        // FYP/vendor/autoload.php (if vendor stays in FYP)
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

function getMerchantName($merchant_id) {
    $conn = dbConnect();
    $stmt = $conn->prepare("SELECT name FROM merchants WHERE merchant_id = ?");
    $stmt->bind_param("s", $merchant_id);
    $stmt->execute();
    $result = $stmt->get_result();
    $conn->close();
    return $result->num_rows > 0 ? $result->fetch_assoc()['name'] : 'Unknown Merchant';
}
// Add this function to your config.php file, after the other email functions

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
?>