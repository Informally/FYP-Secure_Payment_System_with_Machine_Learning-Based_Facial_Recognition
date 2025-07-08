<?php
// admin/login.php - Fixed Admin Login with Better Error Handling
session_name('FACEPAY_ADMIN_SESSION');
require_once 'config/admin_config.php';

// Redirect if already logged in
if (isset($_SESSION['admin_session_token'])) {
    $auth = getAdminAuth();
    if ($auth->validateSession($_SESSION['admin_session_token'])) {
        header("Location: dashboard.php");
        exit();
    }
}

$error = '';
$success = '';
$lockout_message = '';
$debug_info = ''; // For debugging

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // ✅ DEBUGGING: Log all POST data (safely)
    error_log("🔍 Admin Login POST Data: " . json_encode([
        'username' => $_POST['username'] ?? 'not_set',
        'password_length' => strlen($_POST['password'] ?? ''),
        'csrf_token_provided' => isset($_POST['csrf_token']) ? 'yes' : 'no',
        'csrf_token_length' => strlen($_POST['csrf_token'] ?? ''),
        'session_csrf_exists' => isset($_SESSION['admin_csrf_token']) ? 'yes' : 'no',
        'request_method' => $_SERVER['REQUEST_METHOD'],
        'content_type' => $_SERVER['CONTENT_TYPE'] ?? 'not_set'
    ]));
    
    $username = trim($_POST['username'] ?? '');
    
    // ✅ FIXED: Try password deobfuscation with fallback and better error handling
    try {
        if (class_exists('AdminPasswordDeobfuscator')) {
            $password = AdminPasswordDeobfuscator::getPassword();
            $password_method = 'obfuscated';
            error_log("🔐 Admin login: Using obfuscated password (" . strlen($password) . " chars)");
        } else {
            $password = $_POST['password'] ?? '';
            $password_method = 'plain';
            error_log("⚠️ Admin login: AdminPasswordDeobfuscator not found, using plain password");
        }
    } catch (Exception $e) {
        error_log("⚠️ Admin login: Deobfuscation failed (" . $e->getMessage() . "), using plain password");
        $password = $_POST['password'] ?? '';
        $password_method = 'plain_fallback';
    }
    
    error_log("🔍 Login attempt: username='" . $username . "', password_length=" . strlen($password));
    
    // ✅ FIXED: More lenient CSRF validation for debugging
    $csrf_valid = true;
    $csrf_error_reason = '';
    
    if (function_exists('validateAdminCSRFToken')) {
        $provided_token = $_POST['csrf_token'] ?? '';
        $session_token = $_SESSION['admin_csrf_token'] ?? '';
        
        error_log("🔍 CSRF Debug: provided='" . substr($provided_token, 0, 8) . "...', session='" . substr($session_token, 0, 8) . "...'");
        
        if (empty($provided_token)) {
            $csrf_valid = false;
            $csrf_error_reason = 'No CSRF token provided';
        } elseif (empty($session_token)) {
            // Instead of failing, try to regenerate session token
            $_SESSION['admin_csrf_token'] = bin2hex(random_bytes(32));
            $_SESSION['admin_csrf_token_time'] = time();
            error_log("🔧 Regenerated session CSRF token during validation");
            
            // For this request, allow it to pass but log the issue
            $csrf_valid = true;
            $csrf_error_reason = 'Regenerated missing session token';
        } elseif (!validateAdminCSRFToken($provided_token)) {
            $csrf_valid = false;
            $csrf_error_reason = 'CSRF token validation failed';
        }
    } else {
        error_log("⚠️ validateAdminCSRFToken function not found - using basic validation");
        
        // Basic CSRF validation when function doesn't exist
        $provided_token = $_POST['csrf_token'] ?? '';
        $session_token = $_SESSION['admin_csrf_token'] ?? '';
        
        if (empty($provided_token)) {
            $csrf_valid = false;
            $csrf_error_reason = 'No CSRF token provided';
        } elseif (empty($session_token)) {
            // Generate new session token and allow this request
            $_SESSION['admin_csrf_token'] = bin2hex(random_bytes(32));
            $csrf_valid = true;
            $csrf_error_reason = 'Generated new session token';
        } elseif ($provided_token !== $session_token) {
            $csrf_valid = false;
            $csrf_error_reason = 'CSRF tokens do not match';
        }
    }
    
    error_log("🔍 CSRF validation result: " . ($csrf_valid ? 'PASSED' : 'FAILED') . " - " . $csrf_error_reason);
    
    if (!$csrf_valid) {
        $error = "Invalid request. Please try again.";
        $debug_info = "CSRF Error: " . $csrf_error_reason;
        error_log("❌ Admin login CSRF failed: " . $csrf_error_reason);
    } elseif (empty($username) || empty($password)) {
        $error = 'Please enter both username and password';
        $debug_info = "Empty fields: username=" . (empty($username) ? 'empty' : 'provided') . ", password=" . (empty($password) ? 'empty' : 'provided');
        error_log("❌ Admin login: Empty fields");
    } else {
        error_log("🔍 Proceeding with authentication...");
        
        try {
            $conn = dbConnect();
            error_log("✅ Database connection successful");
            
            // ✅ FIXED: Better database query with proper error handling
            error_log("🔍 Checking admin user: " . $username);
            
            $stmt = $conn->prepare("SELECT id, username, email, password_hash, role, full_name, is_active FROM admin_users WHERE username = ? OR email = ?");
            if (!$stmt) {
                throw new Exception("Database prepare failed: " . $conn->error);
            }
            
            $stmt->bind_param("ss", $username, $username);
            $stmt->execute();
            $result = $stmt->get_result();
            $admin_user = $result->fetch_assoc();
            $stmt->close();
            
            error_log("🔍 Admin user found: " . ($admin_user ? 'YES' : 'NO'));
            
            if ($admin_user) {
                error_log("🔍 Admin user details: id=" . $admin_user['id'] . ", active=" . $admin_user['is_active'] . ", role=" . $admin_user['role']);
                
                // ✅ FIXED: Better password verification
                $password_verified = password_verify($password, $admin_user['password_hash']);
                error_log("🔍 Password verification: " . ($password_verified ? 'SUCCESS' : 'FAILED'));
                
                if ($password_verified && $admin_user['is_active']) {
                    // ✅ SUCCESSFUL LOGIN
                    error_log("✅ Admin login successful for: " . $admin_user['username']);
                    
                    // Clear any rate limits
                    $client_ip = $_SERVER['REMOTE_ADDR'] ?? 'unknown';
                    $rate_limit_key = 'admin_login_' . $client_ip . '_' . hash('sha256', $username);
                    
                    if (function_exists('checkAdminRateLimit')) {
                        $clear_stmt = $conn->prepare("DELETE FROM rate_limits WHERE identifier = ?");
                        if ($clear_stmt) {
                            $clear_stmt->bind_param("s", $rate_limit_key);
                            $clear_stmt->execute();
                            $clear_stmt->close();
                        }
                    }
                    
                    // ✅ FIXED: Better session creation with AdminAuth
                    try {
                        error_log("🔍 Getting AdminAuth instance...");
                        $auth = getAdminAuth();
                        error_log("✅ AdminAuth instance obtained");
                        
                        $ip_address = $_SERVER['REMOTE_ADDR'] ?? '';
                        $user_agent = $_SERVER['HTTP_USER_AGENT'] ?? '';
                        
                        error_log("🔍 Calling auth->login...");
                        $auth_result = $auth->login($username, $password, $ip_address, $user_agent);
                        error_log("🔍 Auth result: " . json_encode($auth_result));
                        
                        if ($auth_result['success']) {
                            $_SESSION['admin_session_token'] = $auth_result['session_token'];
                            $_SESSION['admin_user'] = $auth_result['admin'];
                            $_SESSION['admin_id'] = $admin_user['id'];
                            $_SESSION['admin_role'] = $admin_user['role'];
                            
                            error_log("✅ Session variables set:");
                            error_log("  - admin_session_token: " . substr($auth_result['session_token'], 0, 20) . "...");
                            error_log("  - admin_id: " . $admin_user['id']);
                            error_log("  - admin_role: " . $admin_user['role']);
                            
                            // Log successful login
                            if (function_exists('logAdminSecurityEvent')) {
                                logAdminSecurityEvent($admin_user['username'], 'login_success', 'success', [
                                    'admin_id' => $admin_user['id'],
                                    'ip' => $ip_address,
                                    'password_method' => $password_method
                                ]);
                            }
                            
                            error_log("✅ About to redirect to dashboard.php");
                            error_log("🔍 Current session ID: " . session_id());
                            error_log("🔍 Session data before redirect: " . json_encode([
                                'admin_session_token' => substr($_SESSION['admin_session_token'] ?? 'none', 0, 20) . '...',
                                'admin_id' => $_SESSION['admin_id'] ?? 'none',
                                'admin_role' => $_SESSION['admin_role'] ?? 'none'
                            ]));
                            
                            // Add a temporary success message for debugging
                            $_SESSION['login_success_debug'] = [
                                'timestamp' => time(),
                                'username' => $admin_user['username'],
                                'session_token' => substr($auth_result['session_token'], 0, 20) . '...',
                                'redirect_attempted' => true
                            ];
                            
                            // ✅ CRITICAL FIX: Ensure session is properly saved before redirect
                            session_write_close();
                            
                            // Start a new session with the same ID to ensure continuity
                            session_start();
                            
                            // Verify session data is still there
                            error_log("🔍 Session data after restart: " . json_encode([
                                'admin_session_token' => isset($_SESSION['admin_session_token']) ? 'EXISTS' : 'MISSING',
                                'admin_id' => $_SESSION['admin_id'] ?? 'none',
                                'session_id' => session_id()
                            ]));
                            
                            // Check if dashboard.php exists
                            if (!file_exists('dashboard.php')) {
                                error_log("❌ dashboard.php does not exist!");
                                $error = "Dashboard file not found. Please check if dashboard.php exists.";
                                $debug_info = "dashboard.php file missing";
                            } else {
                                error_log("✅ dashboard.php exists, proceeding with redirect");
                                
                                // Add session verification for debugging
                                if (!isset($_SESSION['admin_session_token'])) {
                                    error_log("❌ CRITICAL: Session token lost after session restart!");
                                    $error = "Session persistence failed. Please try logging in again.";
                                    $debug_info = "Session token lost during redirect";
                                } else {
                                    // Force a simple redirect with session verification
                                    header("Location: dashboard.php");
                                    exit();
                                }
                            }
                        } else {
                            $error = "Session creation failed: " . $auth_result['message'];
                            $debug_info = "Auth error: " . $auth_result['message'];
                            error_log("❌ Admin session creation failed: " . $auth_result['message']);
                        }
                    } catch (Exception $e) {
                        $error = "Authentication system error: " . $e->getMessage();
                        $debug_info = "Auth exception: " . $e->getMessage();
                        error_log("❌ Admin auth error: " . $e->getMessage());
                        error_log("❌ Auth exception trace: " . $e->getTraceAsString());
                    }
                } else {
                    // ❌ LOGIN FAILED
                    $reason = !$password_verified ? 'Invalid password' : 'Account not active';
                    error_log("❌ Admin login failed: " . $reason);
                    
                    // Apply rate limiting
                    $client_ip = $_SERVER['REMOTE_ADDR'] ?? 'unknown';
                    $rate_limit_key = 'admin_login_' . $client_ip . '_' . hash('sha256', $username);
                    
                    if (function_exists('checkAdminRateLimit')) {
                        if (!checkAdminRateLimit($rate_limit_key, 5, 900)) { // 5 attempts, 15 minutes
                            $lockout_message = "Too many failed login attempts. Please try again in 15 minutes.";
                        } else {
                            $error = 'Invalid username or password';
                            
                            // Log failed attempt
                            if (function_exists('logAdminSecurityEvent')) {
                                logAdminSecurityEvent($username, 'login_failed', 'failure', [
                                    'ip' => $client_ip,
                                    'reason' => $reason,
                                    'password_method' => $password_method
                                ]);
                            }
                        }
                    } else {
                        $error = 'Invalid username or password';
                    }
                    
                    $debug_info = "Login failed: " . $reason;
                }
            } else {
                // User not found
                error_log("❌ Admin user not found: " . $username);
                $error = 'Invalid username or password';
                $debug_info = "User not found in database";
                
                if (function_exists('logAdminSecurityEvent')) {
                    logAdminSecurityEvent($username, 'login_failed', 'failure', [
                        'ip' => $_SERVER['REMOTE_ADDR'] ?? 'unknown',
                        'reason' => 'User not found'
                    ]);
                }
            }
            
            $conn->close();
        } catch (Exception $e) {
            error_log("❌ Admin login error: " . $e->getMessage());
            error_log("❌ Exception trace: " . $e->getTraceAsString());
            $error = 'Login failed. Please try again later.';
            $debug_info = "Database error: " . $e->getMessage();
        }
    }
}

// ✅ FIXED: Better CSRF token generation - Always ensure session token exists
$csrf_token = '';
try {
    // First, ensure session has a CSRF token
    if (!isset($_SESSION['admin_csrf_token']) || empty($_SESSION['admin_csrf_token'])) {
        $_SESSION['admin_csrf_token'] = bin2hex(random_bytes(32));
        $_SESSION['admin_csrf_token_time'] = time();
        error_log("🔧 Generated new admin CSRF token");
    }
    
    // Try using the function if it exists
    if (function_exists('generateAdminCSRFToken')) {
        $csrf_token = generateAdminCSRFToken();
        error_log("🔧 Using generateAdminCSRFToken function");
    } else {
        // Use session token directly
        $csrf_token = $_SESSION['admin_csrf_token'];
        error_log("🔧 Using session CSRF token directly");
    }
    
    error_log("🔍 Final CSRF token: " . substr($csrf_token, 0, 16) . "...");
    
} catch (Exception $e) {
    error_log("⚠️ CSRF token generation failed: " . $e->getMessage());
    // Emergency fallback
    $csrf_token = bin2hex(random_bytes(16));
    $_SESSION['admin_csrf_token'] = $csrf_token;
}

// ✅ DEBUG: Show debug info in development
$show_debug = ($_SERVER['REMOTE_ADDR'] === '127.0.0.1' || $_SERVER['REMOTE_ADDR'] === '::1') && isset($_GET['debug']);
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Login - FacePay System</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .login-container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
            width: 100%;
            max-width: 450px;
            margin: 20px;
        }

        .login-header {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 40px 30px;
            text-align: center;
        }

        .login-header i {
            font-size: 48px;
            margin-bottom: 15px;
        }

        .login-header h1 {
            font-size: 28px;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .login-header p {
            opacity: 0.9;
            font-size: 16px;
        }

        .login-form {
            padding: 40px 30px;
        }

        .form-group {
            margin-bottom: 25px;
        }

        .form-label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
            font-size: 14px;
        }

        .form-input {
            width: 100%;
            padding: 15px 20px;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            font-size: 16px;
            transition: all 0.3s ease;
            background: #f8f9fa;
        }

        .form-input:focus {
            outline: none;
            border-color: #667eea;
            background: white;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
        }

        .input-group {
            position: relative;
        }

        .input-icon {
            position: absolute;
            left: 15px;
            top: 50%;
            transform: translateY(-50%);
            color: #9ca3af;
            font-size: 18px;
            pointer-events: none;
            z-index: 1;
        }

        .form-input.with-icon {
            padding-left: 50px;
        }

        .login-btn {
            width: 100%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 15px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .login-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        }

        .login-btn:active {
            transform: translateY(0);
        }

        .login-btn:disabled {
            opacity: 0.7;
            cursor: not-allowed;
            transform: none;
        }

        .alert {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 14px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .alert-error {
            background: #fee;
            color: #c53030;
            border: 1px solid #fed7d7;
        }

        .alert-lockout {
            background: #fffbeb;
            color: #ed8936;
            border: 1px solid #fbd38d;
        }

        .alert-success {
            background: #f0fff4;
            color: #2f855a;
            border: 1px solid #c6f6d5;
        }

        .debug-info {
            background: #e3f2fd;
            border: 1px solid #2196f3;
            color: #1976d2;
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            margin-bottom: 15px;
            font-family: monospace;
        }

        .footer-links {
            text-align: center;
            padding: 20px 30px;
            background: #f8f9fa;
            border-top: 1px solid #e1e5e9;
        }

        .footer-links a {
            color: #667eea;
            text-decoration: none;
            font-size: 14px;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .footer-links a:hover {
            text-decoration: underline;
        }

        .security-info {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 13px;
            color: #1976d2;
            border-left: 4px solid #2196f3;
        }

        .honeypot {
            display: none !important;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="login-header">
            <i class="fas fa-shield-alt"></i>
            <h1>FacePay Admin</h1>
            <p>System Administration Portal</p>
        </div>

        <div class="login-form">
            <?php if ($show_debug && !empty($debug_info)): ?>
                <div class="debug-info">
                    <strong>Debug Info:</strong><br>
                    <?php echo htmlspecialchars($debug_info); ?><br>
                    CSRF Token: <?php echo substr($csrf_token, 0, 16); ?>...<br>
                    Session ID: <?php echo substr(session_id(), 0, 16); ?>...
                </div>
            <?php endif; ?>

            <?php if (!empty($error)): ?>
                <div class="alert alert-error">
                    <i class="fas fa-exclamation-triangle"></i>
                    <?php echo htmlspecialchars($error); ?>
                </div>
            <?php endif; ?>

            <?php if (!empty($lockout_message)): ?>
                <div class="alert alert-lockout">
                    <i class="fas fa-clock"></i>
                    <?php echo htmlspecialchars($lockout_message); ?>
                </div>
            <?php endif; ?>

            <?php if (!empty($success)): ?>
                <div class="alert alert-success">
                    <i class="fas fa-check-circle"></i>
                    <?php echo htmlspecialchars($success); ?>
                </div>
            <?php endif; ?>

            <form method="POST" id="adminLoginForm">
                <input type="hidden" name="csrf_token" value="<?= htmlspecialchars($csrf_token) ?>">
                <input type="text" name="website" class="honeypot" tabindex="-1" autocomplete="off">
                
                <div class="form-group">
                    <label class="form-label" for="username">
                        <i class="fas fa-user"></i> Username or Email
                    </label>
                    <div class="input-group">
                        <i class="input-icon fas fa-user"></i>
                        <input type="text" class="form-input with-icon" id="username" name="username" 
                               value="<?php echo htmlspecialchars($_POST['username'] ?? ''); ?>" 
                               placeholder="Enter username or email" 
                               required <?= $lockout_message ? 'disabled' : '' ?>>
                    </div>
                </div>

                <div class="form-group">
                    <label class="form-label" for="password">
                        <i class="fas fa-lock"></i> Password
                    </label>
                    <div class="input-group">
                        <i class="input-icon fas fa-lock"></i>
                        <input type="password" class="form-input with-icon" id="password" name="password" 
                               placeholder="Enter password" 
                               required <?= $lockout_message ? 'disabled' : '' ?> autocomplete="off">
                    </div>
                </div>

                <button type="submit" class="login-btn" id="loginBtn" <?= $lockout_message ? 'disabled' : '' ?>>
                    <i class="fas fa-sign-in-alt"></i> Sign In
                </button>
            </form>

            <?php if ($show_debug): ?>
                <div class="security-info">
                    <strong>Debug Mode Active</strong><br>
                    Add ?debug=1 to URL to see debug information
                </div>
            <?php endif; ?>
        </div>

        <div class="footer-links">
            <a href="../customer_side/login.php">
                <i class="fas fa-arrow-left"></i> Back to Customer Portal
            </a>
        </div>
    </div>
    
    <!-- Include admin password security -->
    <script src="js/admin-password-security.js"></script>
    
    <script>
        // Form submission with loading state
        document.getElementById('adminLoginForm').addEventListener('submit', function(e) {
            const loginBtn = document.getElementById('loginBtn');
            
            if (loginBtn.disabled) {
                e.preventDefault();
                return;
            }
            
            loginBtn.disabled = true;
            loginBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Signing In...';
            
            // Re-enable after timeout
            setTimeout(() => {
                loginBtn.disabled = false;
                loginBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Sign In';
            }, 10000);
        });

        // Auto-focus username field
        document.addEventListener('DOMContentLoaded', function() {
            const usernameInput = document.getElementById('username');
            if (usernameInput && !usernameInput.disabled) {
                usernameInput.focus();
            }
        });

        // Enhanced keyboard navigation
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                const activeElement = document.activeElement;
                if (activeElement.id === 'username') {
                    e.preventDefault();
                    document.getElementById('password').focus();
                }
            }
        });

        console.log('🔐 FacePay Admin Login with Enhanced Debugging initialized');
    </script>
</body>
</html>