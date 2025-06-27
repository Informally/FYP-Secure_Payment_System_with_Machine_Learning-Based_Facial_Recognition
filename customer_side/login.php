<?php
// login.php - Enhanced secure login system
require_once 'config.php';
// Note: aes_encrypt() and aes_decrypt() functions are now included in config.php

// Initialize session management
if (!SessionManager::start()) {
    // Session expired, redirect handled by SessionManager
}

$error = "";
$success = "";
$lockout_message = "";
$attempts_remaining = 0;

// Check if user is already logged in
if (isset($_SESSION['authenticatedUserId'])) {
    header("Location: dashboard.php");
    exit();
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $username = SecurityUtils::sanitizeInput($_POST['username'] ?? '');
    $password = $_POST['password'] ?? '';
    $remember_me = isset($_POST['remember_me']);
    
    // CSRF token validation
    if (!validateCSRFToken($_POST['csrf_token'] ?? '')) {
        $error = "Invalid request. Please try again.";
        SecurityUtils::logSecurityEvent('unknown', 'csrf_validation_failed', 'failure', ['username' => $username]);
    } else {
        // Rate limiting check
        $client_ip = $_SERVER['REMOTE_ADDR'] ?? 'unknown';
        $rate_limit_key = 'login_' . $client_ip . '_' . hash('sha256', $username);
        
        if (!SecurityUtils::checkRateLimit($rate_limit_key, Config::MAX_LOGIN_ATTEMPTS, Config::LOGIN_LOCKOUT_TIME)) {
            $lockout_message = "Too many failed login attempts. Please try again in " . (Config::LOGIN_LOCKOUT_TIME / 60) . " minutes.";
            SecurityUtils::logSecurityEvent($username, 'rate_limit_exceeded', 'failure', ['ip' => $client_ip]);
        } else {
            if (empty($username) || empty($password)) {
                $error = "Please enter both username and password.";
            } else {
                try {
                    $conn = dbConnect(); // Use MySQLi connection
                    
                    // Get user with login attempt tracking (MySQLi style)
                    $stmt = $conn->prepare("
                        SELECT id, user_id, username, password, pin_code, is_verified, account_status, 
                               failed_login_attempts, last_failed_login, locked_until
                        FROM users 
                        WHERE username = ? OR email = ?
                    ");
                    $stmt->bind_param("ss", $username, $username);
                    $stmt->execute();
                    $result = $stmt->get_result();
                    $user = $result->fetch_assoc();
                    
                    if ($user) {
                        // Check if account is locked
                        if ($user['locked_until'] && strtotime($user['locked_until']) > time()) {
                            $lockout_time = strtotime($user['locked_until']) - time();
                            $lockout_message = "Account is temporarily locked. Try again in " . ceil($lockout_time / 60) . " minutes.";
                            SecurityUtils::logSecurityEvent($user['user_id'], 'account_locked_attempt', 'failure');
                        } else {
                            // TEMPORARY FIX - Try both encrypted and plain text comparison
                            $decrypted_password = '';
                            try {
                                $decrypted_password = aes_decrypt($user['password']);
                            } catch (Exception $e) {
                                error_log("Decryption failed: " . $e->getMessage());
                                $decrypted_password = $user['password']; // Try plain text
                            }
                            
                            // Try both encrypted and plain text passwords
                            $password_matches = password_verify($password, $user['password']);
                            
                            // DEBUG - Remove after testing
                            error_log("LOGIN DEBUG - Input: " . $password);
                            error_log("LOGIN DEBUG - Decrypted: " . $decrypted_password);
                            error_log("LOGIN DEBUG - Plain stored: " . $user['password']);
                            error_log("LOGIN DEBUG - Match result: " . ($password_matches ? 'YES' : 'NO'));
                            
                            if ($password_matches) {
                                // Check account status
                                if (!$user['is_verified']) {
                                    $error = "Please verify your email address before logging in.";
                                    SecurityUtils::logSecurityEvent($user['user_id'], 'unverified_login_attempt', 'failure');
                                } elseif ($user['account_status'] === 'suspended') {
                                    $error = "Account is suspended. Please contact support.";
                                    SecurityUtils::logSecurityEvent($user['user_id'], 'suspended_account_login', 'failure');
                                } elseif ($user['account_status'] !== 'active' && $user['account_status'] !== 'pending') {
                                    $error = "Account is not active. Please contact support.";
                                    SecurityUtils::logSecurityEvent($user['user_id'], 'inactive_account_login', 'failure');
                                } else {
                                    // Successful login - reset failed attempts (MySQLi style)
                                    $update_stmt = $conn->prepare("
                                        UPDATE users 
                                        SET failed_login_attempts = 0, 
                                            last_failed_login = NULL, 
                                            locked_until = NULL,
                                            last_login = NOW()
                                        WHERE id = ?
                                    ");
                                    $update_stmt->bind_param("i", $user['id']);
                                    $update_stmt->execute();
                                    
                                    // Set session with basic auth level
                                    SessionManager::setAuthLevel('basic', $user['user_id'] ?? $user['username']);
                                    $_SESSION['username'] = $user['username'];
                                    $_SESSION['user_db_id'] = $user['id'];
                                    $_SESSION['authenticatedUserId'] = $user['user_id'] ?? $user['username'];
                                    
                                    // Handle remember me (simplified for MySQLi)
                                    if ($remember_me) {
                                        $token = bin2hex(random_bytes(32));
                                        $expires = date('Y-m-d H:i:s', time() + (30 * 24 * 60 * 60)); // 30 days
                                        
                                        // Store remember token (check if table exists first)
                                        $check_table = $conn->query("SHOW TABLES LIKE 'remember_tokens'");
                                        if ($check_table->num_rows > 0) {
                                            $token_stmt = $conn->prepare("
                                                INSERT INTO remember_tokens (user_id, token, expires_at) 
                                                VALUES (?, ?, ?)
                                                ON DUPLICATE KEY UPDATE token = ?, expires_at = ?
                                            ");
                                            $token_hash = hash('sha256', $token);
                                            $token_stmt->bind_param("issss", $user['id'], $token_hash, $expires, $token_hash, $expires);
                                            $token_stmt->execute();
                                            
                                            // Set cookie
                                            setcookie('remember_token', $token, time() + (30 * 24 * 60 * 60), '/', '', true, true);
                                        }
                                    }
                                    
                                    SecurityUtils::logSecurityEvent($user['user_id'], 'login_success', 'success', [
                                        'remember_me' => $remember_me,
                                        'user_agent' => $_SERVER['HTTP_USER_AGENT'] ?? 'unknown'
                                    ]);
                                    
                                    // Redirect to dashboard
                                    header("Location: dashboard.php");
                                    exit();
                                }
                            } else {
                                // Failed login - increment counter (MySQLi style)
                                $failed_attempts = $user['failed_login_attempts'] + 1;
                                $lock_account = $failed_attempts >= Config::MAX_LOGIN_ATTEMPTS;
                                
                                $fail_stmt = $conn->prepare("
                                    UPDATE users 
                                    SET failed_login_attempts = ?, 
                                        last_failed_login = NOW(),
                                        locked_until = ?
                                    WHERE id = ?
                                ");
                                
                                $locked_until = $lock_account ? date('Y-m-d H:i:s', time() + Config::LOGIN_LOCKOUT_TIME) : null;
                                $fail_stmt->bind_param("isi", $failed_attempts, $locked_until, $user['id']);
                                $fail_stmt->execute();
                                
                                if ($lock_account) {
                                    $lockout_message = "Account locked due to too many failed attempts. Try again in " . (Config::LOGIN_LOCKOUT_TIME / 60) . " minutes.";
                                    SecurityUtils::logSecurityEvent($user['user_id'], 'account_locked', 'failure');
                                } else {
                                    $attempts_remaining = Config::MAX_LOGIN_ATTEMPTS - $failed_attempts;
                                    $error = "Invalid credentials. $attempts_remaining attempts remaining.";
                                    SecurityUtils::logSecurityEvent($user['user_id'], 'login_failed', 'failure');
                                }
                            }
                        }
                    } else {
                        $error = "Invalid username or password.";
                        SecurityUtils::logSecurityEvent($username, 'login_user_not_found', 'failure');
                    }
                    
                    $conn->close();
                    
                } catch (Exception $e) {
                    error_log("Login error: " . $e->getMessage());
                    $error = "Login failed. Please try again later.";
                }
            }
        }
    }
}

// Check for remember me token (simplified for MySQLi)
if (isset($_COOKIE['remember_token']) && !isset($_SESSION['authenticatedUserId'])) {
    try {
        $conn = dbConnect();
        $token_hash = hash('sha256', $_COOKIE['remember_token']);
        
        // Check if remember_tokens table exists
        $check_table = $conn->query("SHOW TABLES LIKE 'remember_tokens'");
        if ($check_table->num_rows > 0) {
            $stmt = $conn->prepare("
                SELECT u.id, u.user_id, u.username 
                FROM users u
                JOIN remember_tokens rt ON u.id = rt.user_id
                WHERE rt.token = ? AND rt.expires_at > NOW() AND (u.account_status = 'active' OR u.account_status = 'pending')
            ");
            $stmt->bind_param("s", $token_hash);
            $stmt->execute();
            $result = $stmt->get_result();
            $user = $result->fetch_assoc();
            
            if ($user) {
                // Auto-login user
                SessionManager::setAuthLevel('basic', $user['user_id'] ?? $user['username']);
                $_SESSION['username'] = $user['username'];
                $_SESSION['user_db_id'] = $user['id'];
                $_SESSION['authenticatedUserId'] = $user['user_id'] ?? $user['username'];
                
                SecurityUtils::logSecurityEvent($user['user_id'], 'remember_token_login', 'success');
                
                header("Location: dashboard.php");
                exit();
            } else {
                // Invalid or expired token
                setcookie('remember_token', '', time() - 3600, '/', '', true, true);
            }
        }
        $conn->close();
    } catch (Exception $e) {
        error_log("Remember token error: " . $e->getMessage());
    }
}

// Generate CSRF token
$csrf_token = generateCSRFToken();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Secure FacePay System</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --primary: #3498db;
            --primary-light: #e3f2fd;
            --primary-dark: #2980b9;
            --secondary: #f8f9fa;
            --text-dark: #2d3748;
            --text-medium: #4a5568;
            --text-light: #718096;
            --border: #e2e8f0;
            --error: #e53e3e;
            --success: #38a169;
            --warning: #ed8936;
            --shadow: rgba(0,0,0,0.1);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: var(--text-dark);
            line-height: 1.5;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            width: 100%;
            max-width: 420px;
        }
        
        .form-container {
            background-color: white;
            border-radius: 16px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            padding: 40px 35px;
            animation: fadeUp 0.6s ease-out;
            position: relative;
            overflow: hidden;
        }
        
        .form-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--primary), #667eea);
        }
        
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .logo-section {
            text-align: center;
            margin-bottom: 35px;
        }
        
        .logo {
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, var(--primary), #667eea);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        .logo i {
            font-size: 32px;
            color: white;
        }
        
        h2 {
            color: var(--text-dark);
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        
        .subtitle {
            color: var(--text-light);
            font-size: 14px;
        }
        
        .form-group {
            margin-bottom: 24px;
        }
        
        label {
            display: block;
            font-size: 14px;
            font-weight: 600;
            color: var(--text-medium);
            margin-bottom: 8px;
        }
        
        .input-group {
            position: relative;
        }
        
        .input-group input {
            width: 100%;
            height: 48px;
            padding: 0 20px 0 50px;
            border: 2px solid var(--border);
            border-radius: 12px;
            font-size: 15px;
            transition: all 0.3s ease;
            color: var(--text-dark);
            background-color: #fafafa;
        }
        
        .input-group input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(52, 152, 219, 0.1);
            outline: none;
            background-color: white;
        }
        
        .input-group input::placeholder {
            color: #a0aec0;
        }
        
        .input-icon {
            position: absolute;
            left: 18px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-light);
            font-size: 16px;
            pointer-events: none;
            transition: color 0.3s ease;
        }
        
        .input-group input:focus + .input-icon {
            color: var(--primary);
        }
        
        .form-options {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            font-size: 14px;
        }
        
        .remember-me {
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--text-medium);
        }
        
        .remember-me input[type="checkbox"] {
            width: 16px;
            height: 16px;
            accent-color: var(--primary);
        }
        
        .forgot-password {
            color: var(--primary);
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s ease;
        }
        
        .forgot-password:hover {
            color: var(--primary-dark);
            text-decoration: underline;
        }
        
        .btn {
            width: 100%;
            height: 50px;
            background: linear-gradient(135deg, var(--primary), #667eea);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            margin-bottom: 25px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(52, 152, 219, 0.3);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .message {
            padding: 15px 18px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 500;
        }
        
        .error-message {
            background-color: #fff5f5;
            color: var(--error);
            border: 1px solid #fed7d7;
        }
        
        .lockout-message {
            background-color: #fffbeb;
            color: var(--warning);
            border: 1px solid #fbd38d;
        }
        
        .success-message {
            background-color: #f0fff4;
            color: var(--success);
            border: 1px solid #c6f6d5;
        }
        
        .form-footer {
            text-align: center;
            color: var(--text-light);
            font-size: 14px;
        }
        
        .form-footer a {
            color: var(--primary);
            text-decoration: none;
            font-weight: 600;
            transition: color 0.3s ease;
        }
        
        .form-footer a:hover {
            color: var(--primary-dark);
            text-decoration: underline;
        }
        
        .security-info {
            background-color: var(--primary-light);
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 13px;
            color: var(--text-medium);
            border-left: 4px solid var(--primary);
        }
        
        .alternative-auth {
            text-align: center;
            margin-top: 25px;
            padding-top: 25px;
            border-top: 1px solid var(--border);
        }
        
        .alt-btn {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 12px 24px;
            background-color: #f8f9fa;
            color: var(--text-medium);
            text-decoration: none;
            border-radius: 10px;
            font-weight: 500;
            transition: all 0.3s ease;
            border: 1px solid var(--border);
        }
        
        .alt-btn:hover {
            background-color: var(--primary);
            color: white;
            transform: translateY(-1px);
        }
        
        @media (max-width: 480px) {
            .form-container {
                padding: 30px 25px;
            }
            
            .form-options {
                flex-direction: column;
                gap: 15px;
                align-items: flex-start;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="form-container">
            <div class="logo-section">
                <div class="logo">
                    <i class="fas fa-user-shield"></i>
                </div>
                <h2>Welcome Back</h2>
                <p class="subtitle">Sign in to your secure FacePay account</p>
            </div>
            
            <?php if ($error): ?>
                <div class="message error-message">
                    <i class="fas fa-exclamation-circle"></i>
                    <?= htmlspecialchars($error) ?>
                </div>
            <?php endif; ?>
            
            <?php if ($lockout_message): ?>
                <div class="message lockout-message">
                    <i class="fas fa-clock"></i>
                    <?= htmlspecialchars($lockout_message) ?>
                </div>
            <?php endif; ?>
            
            <?php if ($success): ?>
                <div class="message success-message">
                    <i class="fas fa-check-circle"></i>
                    <?= htmlspecialchars($success) ?>
                </div>
            <?php endif; ?>
            
            <form method="POST" id="loginForm">
                <input type="hidden" name="csrf_token" value="<?= $csrf_token ?>">
                
                <div class="form-group">
                    <label for="username">Username or Email</label>
                    <div class="input-group">
                        <input type="text" id="username" name="username" 
                               placeholder="Enter your username or email" 
                               value="<?= isset($_POST['username']) ? htmlspecialchars($_POST['username']) : '' ?>" 
                               required <?= $lockout_message ? 'disabled' : '' ?>>
                        <i class="fas fa-user input-icon"></i>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="password">Password</label>
                    <div class="input-group">
                        <input type="password" id="password" name="password" 
                               placeholder="Enter your password" 
                               required <?= $lockout_message ? 'disabled' : '' ?>>
                        <i class="fas fa-lock input-icon"></i>
                    </div>
                </div>
                
                <div class="form-options">
                    <label class="remember-me">
                        <input type="checkbox" name="remember_me" value="1" <?= $lockout_message ? 'disabled' : '' ?>>
                        Remember me
                    </label>
                    <a href="forgot_password.php" class="forgot-password">Forgot password?</a>
                </div>
                
                <button type="submit" class="btn" <?= $lockout_message ? 'disabled' : '' ?>>
                    <i class="fas fa-sign-in-alt"></i>
                    Sign In
                </button>
            </form>
            
            <div class="alternative-auth">
                <p style="margin-bottom: 15px; color: var(--text-light);">Or continue with</p>
                <a href="scan.php" class="alt-btn">
                    <i class="fas fa-camera"></i>
                    Facial Recognition
                </a>
            </div>
            
            <div class="form-footer">
                <p>Don't have an account? <a href="register.php">Create one here</a></p>
            </div>
            
            <div class="security-info">
                <i class="fas fa-shield-alt"></i>
                Your login is protected with advanced security measures including rate limiting and session management.
            </div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const form = document.getElementById('loginForm');
            const submitBtn = form.querySelector('.btn');
            const usernameInput = document.getElementById('username');
            const passwordInput = document.getElementById('password');
            
            // Focus first empty field
            if (!usernameInput.value) {
                usernameInput.focus();
            } else if (!passwordInput.value) {
                passwordInput.focus();
            }
            
            // Form submission handling
            form.addEventListener('submit', function(e) {
                if (submitBtn.disabled) {
                    e.preventDefault();
                    return;
                }
                
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Signing In...';
                
                // Re-enable after 3 seconds to prevent permanent lockout
                setTimeout(() => {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Sign In';
                }, 3000);
            });
            
            // Input validation
            const inputs = [usernameInput, passwordInput];
            inputs.forEach(input => {
                input.addEventListener('input', function() {
                    if (this.value.trim()) {
                        this.style.borderColor = 'var(--success)';
                    } else {
                        this.style.borderColor = 'var(--border)';
                    }
                });
            });
        });
    </script>
</body>
</html>