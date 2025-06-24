<?php
// register.php - Enhanced Registration System with Database Integration
require_once 'config.php';
// Note: aes_encrypt() and aes_decrypt() functions are now included in config.php

// Check if user is already logged in
SessionManager::start();
if (isset($_SESSION['authenticatedUserId'])) {
    header("Location: dashboard.php");
    exit();
}

$error = "";
$success = "";
$username_error = false;
$email_error = false;

// Save form data to repopulate fields on error
$form_data = [
    'username' => '',
    'full_name' => '',
    'email' => ''
];

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // CSRF validation
    if (!validateCSRFToken($_POST['csrf_token'] ?? '')) {
        $error = "Invalid request. Please try again.";
        SecurityUtils::logSecurityEvent('unknown', 'csrf_validation_failed', 'failure', ['action' => 'register']);
    } else {
        // Rate limiting check
        $client_ip = $_SERVER['REMOTE_ADDR'] ?? 'unknown';
        if (!SecurityUtils::checkRateLimit('register_' . $client_ip, 5, 3600)) { // 5 attempts per hour
            $error = "Too many registration attempts. Please try again in an hour.";
            SecurityUtils::logSecurityEvent('unknown', 'registration_rate_limit_exceeded', 'failure', ['ip' => $client_ip]);
        } else {
            try {
                $conn = dbConnect(); // Use MySQLi connection to match config.php
                
                $username = SecurityUtils::sanitizeInput($_POST['username'] ?? '');
                $full_name = SecurityUtils::sanitizeInput($_POST['full_name'] ?? '');
                $email = SecurityUtils::sanitizeInput($_POST['email'] ?? '');
                $password = $_POST['password'] ?? '';
                $pin = implode('', $_POST['pin'] ?? []);

                // Store form data to repopulate on error
                $form_data = [
                    'username' => $username,
                    'full_name' => $full_name,
                    'email' => $email
                ];

                // Validation
                if (empty($username) || empty($full_name) || empty($email) || empty($password) || empty($pin)) {
                    $error = "All fields are required.";
                } elseif (strlen($username) < 3) {
                    $error = "Username must be at least 3 characters long.";
                    $username_error = true;
                } elseif (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
                    $error = "Please enter a valid email address.";
                    $email_error = true;
                } elseif (!preg_match('/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^a-zA-Z\d]).{8,}$/', $password)) {
                    $error = "Password must be at least 8 characters and include uppercase, lowercase, number, and symbol.";
                } elseif (!preg_match('/^\d{6}$/', $pin)) {
                    $error = "PIN must be exactly 6 digits.";
                } else {
                    // Check if username already exists (MySQLi style)
                    $stmt = $conn->prepare("SELECT id FROM users WHERE username = ?");
                    $stmt->bind_param("s", $username);
                    $stmt->execute();
                    $result = $stmt->get_result();

                    if ($result->num_rows > 0) {
                        $error = "Username is already taken. Please choose a different username.";
                        $username_error = true;
                        SecurityUtils::logSecurityEvent($username, 'registration_duplicate_username', 'failure');
                    } else {
                        // Check if email already exists (MySQLi style)
                        $stmt = $conn->prepare("SELECT id FROM users WHERE email = ?");
                        $stmt->bind_param("s", $email);
                        $stmt->execute();
                        $result = $stmt->get_result();

                        if ($result->num_rows > 0) {
                            $error = "Email is already registered. Please use a different email address.";
                            $email_error = true;
                            SecurityUtils::logSecurityEvent($email, 'registration_duplicate_email', 'failure');
                        } else {
                            // Generate unique user_id and verification token
                            $user_id = uniqid('USER_', true);
                            $verification_token = bin2hex(random_bytes(32));
                            
                            // Encrypt sensitive data
                            $encrypted_password = aes_encrypt($password);
                            $encrypted_pin = aes_encrypt($pin);

                            // Insert user with MySQLi (matching your actual database structure)
                            $stmt = $conn->prepare("INSERT INTO users (user_id, username, full_name, email, password, pin_code, verification_token, is_verified) VALUES (?, ?, ?, ?, ?, ?, ?, 0)");
                            $stmt->bind_param("sssssss", $user_id, $username, $full_name, $email, $encrypted_password, $encrypted_pin, $verification_token);

                            if ($stmt->execute()) {
                                $new_user_db_id = $conn->insert_id; // MySQLi method

                                // Create wallet (simplified like your working code)
                                $welcome_bonus = '50.00';
                                $wallet_stmt = $conn->prepare("INSERT INTO wallets (user_id, balance) VALUES (?, ?)");
                                $wallet_stmt->bind_param("is", $new_user_db_id, $welcome_bonus);
                                $wallet_stmt->execute();

                                // Log welcome bonus transaction with appropriate details
                                $transaction_id = 'BONUS_' . strtoupper(uniqid());
                                $log_stmt = $conn->prepare("
                                    INSERT INTO transactions (user_id, transaction_id, amount, status, transaction_type, payment_method) 
                                    VALUES (?, ?, ?, 'success', 'Welcome Bonus', 'System')
                                ");
                                $log_stmt->bind_param("isd", $new_user_db_id, $transaction_id, $welcome_bonus);
                                $log_stmt->execute();

                                // Send verification email using the function from config.php
                                if (sendVerificationEmail($email, $verification_token)) {
                                    SecurityUtils::logSecurityEvent($user_id, 'registration_success', 'success', [
                                        'username' => $username,
                                        'email' => hash('sha256', $email), // Hash email for privacy
                                        'welcome_bonus' => $welcome_bonus
                                    ]);

                                    $success = "Registration successful! Please check your email to verify your account. Your account will be activated and the welcome bonus of RM " . $welcome_bonus . " will be added to your wallet only after email verification.";
                                    
                                    // Clear form data on success
                                    $form_data = [
                                        'username' => '',
                                        'full_name' => '',
                                        'email' => ''
                                    ];
                                } else {
                                    $error = "Registration successful, but verification email could not be sent. Please contact support.";
                                    SecurityUtils::logSecurityEvent($user_id, 'registration_email_failed', 'warning');
                                }
                            } else {
                                $error = "Registration failed. Please try again later.";
                                SecurityUtils::logSecurityEvent($username, 'registration_database_error', 'failure');
                            }
                        }
                    }
                }
            } catch (Exception $e) {
                error_log("Registration error: " . $e->getMessage());
                $error = "An error occurred during registration. Please try again later.";
                SecurityUtils::logSecurityEvent('unknown', 'registration_system_error', 'failure', [
                    'error' => $e->getMessage()
                ]);
            }
        }
    }
}

$csrf_token = generateCSRFToken();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Register - Secure FacePay</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --primary: #3498db;
            --primary-light: #e3f2fd;
            --primary-dark: #2980b9;
            --secondary: #f8f9fa;
            --success: #38a169;
            --warning: #ed8936;
            --error: #e53e3e;
            --text-dark: #2d3748;
            --text-medium: #4a5568;
            --text-light: #718096;
            --border: #e2e8f0;
            --bg-light: #f7fafc;
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
            max-width: 500px;
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
            margin-bottom: 30px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group:last-of-type {
            margin-bottom: 25px;
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
        
        /* Error state */
        .input-error {
            border-color: var(--error) !important;
        }
        
        .input-error:focus {
            box-shadow: 0 0 0 4px rgba(229, 62, 62, 0.1) !important;
        }
        
        .field-error {
            color: var(--error);
            font-size: 12px;
            margin-top: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        /* Password Strength */
        .password-strength {
            height: 6px;
            border-radius: 3px;
            background-color: #edf2f7;
            margin-top: 8px;
            overflow: hidden;
        }
        
        .password-strength-meter {
            height: 100%;
            width: 0;
            border-radius: 3px;
            transition: all 0.3s;
        }
        
        .strength-text {
            font-size: 12px;
            margin-top: 6px;
            text-align: right;
            color: var(--text-medium);
            font-weight: 500;
        }
        
        /* Password Requirements */
        .password-requirements {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin-top: 12px;
        }
        
        .requirement {
            display: flex;
            align-items: center;
            color: var(--text-light);
            font-size: 12px;
        }
        
        .requirement-indicator {
            display: inline-block;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            margin-right: 8px;
            background-color: #cbd5e0;
            transition: background-color 0.3s;
        }
        
        .requirement.met .requirement-indicator {
            background-color: var(--success);
        }
        
        .requirement.met {
            color: var(--success);
        }
        
        /* PIN Input */
        .pin-container {
            margin-top: 8px;
        }
        
        .pin-box {
            display: flex;
            gap: 10px;
            justify-content: space-between;
        }
        
        .pin-box input {
            width: 45px;
            height: 45px;
            text-align: center;
            font-size: 18px;
            font-weight: 600;
            border: 2px solid var(--border);
            border-radius: 8px;
            -webkit-appearance: none;
            -moz-appearance: textfield;
            background-color: #fafafa;
            transition: all 0.3s ease;
        }
        
        .pin-box input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
            outline: none;
            background-color: white;
        }
        
        /* Messages */
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
        
        .success-message {
            background-color: #f0fff4;
            color: var(--success);
            border: 1px solid #c6f6d5;
        }
        
        /* Button */
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
        
        /* Form Footer */
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
        
        /* Security Info */
        .security-info {
            background-color: var(--primary-light);
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 13px;
            color: var(--text-medium);
            border-left: 4px solid var(--primary);
        }
        
        .security-info i {
            color: var(--primary);
            margin-right: 8px;
        }
        
        /* Responsive Design */
        @media (max-width: 600px) {
            .container {
                padding: 15px;
            }
            
            .form-container {
                padding: 30px 25px;
            }
            
            .password-requirements {
                grid-template-columns: 1fr;
            }
            
            .pin-box {
                gap: 8px;
            }
            
            .pin-box input {
                width: 40px;
                height: 40px;
                font-size: 16px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="form-container">
            <div class="logo-section">
                <div class="logo">
                    <i class="fas fa-user-plus"></i>
                </div>
                <h2>Create Account</h2>
                <p class="subtitle">Join Secure FacePay for safe and convenient payments</p>
            </div>
            
            <?php if ($error): ?>
                <div class="message error-message">
                    <i class="fas fa-exclamation-circle"></i>
                    <?= htmlspecialchars($error) ?>
                </div>
            <?php endif; ?>
            
            <?php if ($success): ?>
                <div class="message success-message">
                    <i class="fas fa-check-circle"></i>
                    <?= htmlspecialchars($success) ?>
                </div>
            <?php endif; ?>

            <form method="POST" id="registrationForm">
                <input type="hidden" name="csrf_token" value="<?= $csrf_token ?>">
                
                <!-- Full Name -->
                <div class="form-group">
                    <label for="full_name">Full Name</label>
                    <div class="input-group">
                        <input type="text" id="full_name" name="full_name" 
                               placeholder="Enter your full name" 
                               value="<?= htmlspecialchars($form_data['full_name']) ?>" required>
                        <i class="fas fa-user input-icon"></i>
                    </div>
                </div>
                
                <!-- Username -->
                <div class="form-group">
                    <label for="username">Username</label>
                    <div class="input-group">
                        <input type="text" id="username" name="username" 
                               placeholder="Choose a unique username" 
                               value="<?= htmlspecialchars($form_data['username']) ?>" 
                               class="<?= $username_error ? 'input-error' : '' ?>" required>
                        <i class="fas fa-at input-icon"></i>
                    </div>
                    <?php if ($username_error): ?>
                        <div class="field-error">
                            <i class="fas fa-exclamation-circle"></i>
                            This username is already taken
                        </div>
                    <?php endif; ?>
                </div>
                
                <!-- Email -->
                <div class="form-group">
                    <label for="email">Email Address</label>
                    <div class="input-group">
                        <input type="email" id="email" name="email" 
                               placeholder="Enter your email address" 
                               value="<?= htmlspecialchars($form_data['email']) ?>" 
                               class="<?= $email_error ? 'input-error' : '' ?>" required>
                        <i class="fas fa-envelope input-icon"></i>
                    </div>
                    <?php if ($email_error): ?>
                        <div class="field-error">
                            <i class="fas fa-exclamation-circle"></i>
                            This email is already registered
                        </div>
                    <?php endif; ?>
                </div>
                
                <!-- Password -->
                <div class="form-group">
                    <label for="password">Password</label>
                    <div class="input-group">
                        <input type="password" id="password" name="password" 
                               placeholder="Create a strong password" required>
                        <i class="fas fa-lock input-icon"></i>
                    </div>
                    
                    <div class="password-strength">
                        <div class="password-strength-meter" id="password-meter"></div>
                    </div>
                    <div class="strength-text" id="password-strength-text"></div>
                    
                    <div class="password-requirements">
                        <div class="requirement" id="req-length">
                            <span class="requirement-indicator"></span> 8+ characters
                        </div>
                        <div class="requirement" id="req-lowercase">
                            <span class="requirement-indicator"></span> Lowercase (a-z)
                        </div>
                        <div class="requirement" id="req-uppercase">
                            <span class="requirement-indicator"></span> Uppercase (A-Z)
                        </div>
                        <div class="requirement" id="req-number">
                            <span class="requirement-indicator"></span> Number (0-9)
                        </div>
                        <div class="requirement" id="req-special">
                            <span class="requirement-indicator"></span> Special (!@#$%)
                        </div>
                    </div>
                </div>
                
                <!-- PIN -->
                <div class="form-group">
                    <label for="pin">6-digit PIN for Transactions</label>
                    <div class="pin-container">
                        <div class="pin-box">
                            <?php for ($i = 0; $i < 6; $i++): ?>
                                <input type="password" name="pin[]" maxlength="1" pattern="\d" inputmode="numeric" required>
                            <?php endfor; ?>
                        </div>
                    </div>
                </div>
                
                <!-- Submit Button -->
                <button type="submit" class="btn" id="submitBtn">
                    <i class="fas fa-user-plus"></i> Create Account
                </button>
            </form>
            
            <div class="form-footer">
                <p>Already have an account? <a href="scan.php">Sign In with Face Recognition</a> or <a href="login.php">Use Username & Password</a></p>
            </div>
            
            <div class="security-info">
                <i class="fas fa-shield-alt"></i>
                Your data is protected with enterprise-level encryption. <strong>Important:</strong> You must verify your email address to activate your account and receive the RM 50.00 welcome bonus. Check your inbox after registration!
            </div>
        </div>
    </div>

    <script>
        // PIN input handling
        const pinInputs = document.querySelectorAll(".pin-box input");
        
        // Focus the first input on page load
        setTimeout(() => {
            if (document.activeElement.tagName !== 'INPUT') {
                document.getElementById('full_name').focus();
            }
        }, 500);
        
        // PIN input event handlers
        pinInputs.forEach((input, index) => {
            // Handle input
            input.addEventListener("input", (e) => {
                // Only allow numbers
                e.target.value = e.target.value.replace(/[^0-9]/g, '');
                
                // Move to next input when current one is filled
                if (e.target.value.length === 1 && index < pinInputs.length - 1) {
                    pinInputs[index + 1].focus();
                }
            });
            
            // Handle backspace
            input.addEventListener("keydown", (e) => {
                if (e.key === "Backspace") {
                    if (e.target.value === '' && index > 0) {
                        pinInputs[index - 1].focus();
                        pinInputs[index - 1].value = '';
                        e.preventDefault();
                    }
                }
                
                // Arrow key navigation
                if (e.key === "ArrowLeft" && index > 0) {
                    pinInputs[index - 1].focus();
                    e.preventDefault();
                }
                
                if (e.key === "ArrowRight" && index < pinInputs.length - 1) {
                    pinInputs[index + 1].focus();
                    e.preventDefault();
                }
            });
            
            // Handle paste
            input.addEventListener("paste", (e) => {
                e.preventDefault();
                const pasteData = e.clipboardData.getData('text').replace(/[^0-9]/g, '').slice(0, 6);
                
                // Fill inputs with pasted data
                for (let i = 0; i < Math.min(pasteData.length, pinInputs.length); i++) {
                    pinInputs[i].value = pasteData[i];
                }
                
                // Focus appropriate input after paste
                if (pasteData.length >= pinInputs.length) {
                    pinInputs[pinInputs.length - 1].focus();
                } else if (pasteData.length > 0) {
                    pinInputs[pasteData.length].focus();
                }
            });
        });
        
        // Password strength meter and requirements
        const passwordInput = document.getElementById('password');
        const strengthMeter = document.getElementById('password-meter');
        const strengthText = document.getElementById('password-strength-text');
        
        // Requirement elements
        const reqLength = document.getElementById('req-length');
        const reqLowercase = document.getElementById('req-lowercase');
        const reqUppercase = document.getElementById('req-uppercase');
        const reqNumber = document.getElementById('req-number');
        const reqSpecial = document.getElementById('req-special');
        
        passwordInput.addEventListener('input', updatePasswordStrength);
        
        function updatePasswordStrength() {
            const password = passwordInput.value;
            let strength = 0;
            let comment = '';
            
            // Check requirements
            const hasLength = password.length >= 8;
            const hasLowercase = /[a-z]/.test(password);
            const hasUppercase = /[A-Z]/.test(password);
            const hasNumber = /[0-9]/.test(password);
            const hasSpecial = /[^a-zA-Z0-9]/.test(password);
            
            // Update requirement UI
            updateRequirement(reqLength, hasLength);
            updateRequirement(reqLowercase, hasLowercase);
            updateRequirement(reqUppercase, hasUppercase);
            updateRequirement(reqNumber, hasNumber);
            updateRequirement(reqSpecial, hasSpecial);
            
            // Calculate strength
            if (hasLength) strength += 20;
            if (hasLowercase) strength += 20;
            if (hasUppercase) strength += 20;
            if (hasNumber) strength += 20;
            if (hasSpecial) strength += 20;
            
            // Update visual elements
            strengthMeter.style.width = strength + '%';
            
            // Set color based on strength
            if (password.length === 0) {
                strengthMeter.style.backgroundColor = '#edf2f7';
                comment = '';
            } else if (strength <= 20) {
                strengthMeter.style.backgroundColor = '#e53e3e';
                comment = 'Very weak';
            } else if (strength <= 40) {
                strengthMeter.style.backgroundColor = '#ed8936';
                comment = 'Weak';
            } else if (strength <= 60) {
                strengthMeter.style.backgroundColor = '#ecc94b';
                comment = 'Fair';
            } else if (strength <= 80) {
                strengthMeter.style.backgroundColor = '#48bb78';
                comment = 'Good';
            } else {
                strengthMeter.style.backgroundColor = '#38a169';
                comment = 'Strong';
            }
            
            strengthText.textContent = comment;
        }
        
        function updateRequirement(element, isMet) {
            if (isMet) {
                element.classList.add('met');
            } else {
                element.classList.remove('met');
            }
        }
    </script>
</body>
</html>