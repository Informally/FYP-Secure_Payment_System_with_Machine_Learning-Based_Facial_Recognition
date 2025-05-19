<?php
require_once 'config.php';
require_once 'encrypt.php';
require_once 'vendor/autoload.php';
use PHPMailer\PHPMailer\PHPMailer;
use PHPMailer\PHPMailer\Exception;

function sendVerificationEmail($email, $token) {
    $mail = new PHPMailer(true);
    try {
        $mail->isSMTP();
        $mail->Host = 'smtp.gmail.com';
        $mail->SMTPAuth = true;
        $mail->Username = 'authorizedpay88@gmail.com';
        $mail->Password = 'tbpwxivpydxjqrwq';
        $mail->SMTPSecure = 'tls';
        $mail->Port = 587;

        $mail->setFrom('authorizedpay88@gmail.com', 'Facial Pay System');
        $mail->addAddress($email);
        $mail->isHTML(true);
        $mail->Subject = 'Verify your email';
        $mail->Body = "Click the link below to verify your account:<br><a href='http://localhost/FYP/verify.php?token=$token'>Verify Account</a>";
        $mail->send();
    } catch (Exception $e) {
        error_log("Mailer Error: {$mail->ErrorInfo}");
    }
}

session_start();
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
    $conn = dbConnect();

    $username = trim($_POST['username']);
    $full_name = trim($_POST['full_name']);
    $email = trim($_POST['email']);
    $password = $_POST['password'];
    $pin = implode('', $_POST['pin'] ?? []);

    // Store form data to repopulate on error
    $form_data = [
        'username' => $username,
        'full_name' => $full_name,
        'email' => $email
    ];

    $password_valid = preg_match('/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^a-zA-Z\d]).{8,}$/', $password);

    if (empty($username) || empty($full_name) || empty($email) || empty($password) || empty($pin)) {
        $error = "All fields are required.";
    } elseif (!$password_valid) {
        $error = "Password must be at least 8 characters and include uppercase, lowercase, number, and symbol.";
    } elseif (!preg_match('/^\d{6}$/', $pin)) {
        $error = "PIN must be exactly 6 digits.";
    } else {
        // First check if username already exists
        $stmt = $conn->prepare("SELECT id FROM users WHERE username = ?");
        $stmt->bind_param("s", $username);
        $stmt->execute();
        $result = $stmt->get_result();

        if ($result->num_rows > 0) {
            $error = "Username is already taken. Please choose a different username.";
            $username_error = true;
        } else {
            // Then check if email already exists
            $stmt = $conn->prepare("SELECT id FROM users WHERE email = ?");
            $stmt->bind_param("s", $email);
            $stmt->execute();
            $result = $stmt->get_result();

            if ($result->num_rows > 0) {
                $error = "Email is already registered. Please use a different email address.";
                $email_error = true;
            } else {
                $encrypted_password = aes_encrypt($password);
                $encrypted_pin = aes_encrypt($pin);
                $token = bin2hex(random_bytes(16));

                $stmt = $conn->prepare("INSERT INTO users (username, full_name, email, password, pin_code, verification_token, is_verified) VALUES (?, ?, ?, ?, ?, ?, 0)");
                $stmt->bind_param("ssssss", $username, $full_name, $email, $encrypted_password, $encrypted_pin, $token);

                if ($stmt->execute()) {
                    $user_id = $stmt->insert_id;

                    $wallet = $conn->prepare("INSERT INTO wallets (user_id, balance) VALUES (?, 50.00)");
                    $wallet->bind_param("i", $user_id);
                    $wallet->execute();

                    $log = $conn->prepare("INSERT INTO transactions (user_id, amount, status) VALUES (?, 50.00, 'success')");
                    $log->bind_param("i", $user_id);
                    $log->execute();

                    sendVerificationEmail($email, $token);
                    $success = "Registration successful! Please check your email to verify your account.";
                    
                    // Clear form data on success
                    $form_data = [
                        'username' => '',
                        'full_name' => '',
                        'email' => ''
                    ];
                } else {
                    $error = "Error occurred. Please try again later.";
                }
            }
        }
    }
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Register - Facial Pay System</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link rel="stylesheet" href="style.css">
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
            background-color: #f0f4f8;
            color: var(--text-dark);
            line-height: 1.5;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 12px;
        }
        
        .container {
            width: 100%;
            max-width: 480px;
        }
        
        .form-container {
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            padding: 25px 30px;
            animation: fadeUp 0.4s ease-out;
        }
        
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        h2 {
            color: var(--text-dark);
            font-size: 22px;
            font-weight: 600;
            margin-bottom: 20px;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        
        h2 i {
            color: var(--primary);
        }
        
        .form-group {
            margin-bottom: 18px;
        }
        
        .form-group:last-of-type {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            font-size: 14px;
            font-weight: 500;
            color: var(--text-medium);
            margin-bottom: 6px;
        }
        
        .input-group {
            position: relative;
        }
        
        .input-group input {
            padding-left: 38px;
            width: 100%;
            height: 42px;
            border-radius: 8px;
            border: 1px solid var(--border);
            font-size: 14px;
            transition: all 0.2s;
            color: var(--text-dark);
        }
        
        .input-group input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.15);
            outline: none;
        }
        
        .input-group input::placeholder {
            color: #a0aec0;
        }
        
        .input-icon {
            position: absolute;
            left: 14px;
            top: 50%;
            transform: translateY(-50%);
            color: #a0aec0;
            pointer-events: none;
        }
        
        /* Password Strength */
        .password-strength {
            height: 5px;
            border-radius: 3px;
            background-color: #edf2f7;
            margin-top: 6px;
            overflow: hidden;
        }
        
        .password-strength-meter {
            height: 100%;
            width: 0;
            border-radius: 3px;
            transition: all 0.3s;
        }
        
        .strength-text {
            font-size: 11px;
            margin-top: 3px;
            text-align: right;
            color: var(--text-medium);
        }
        
        /* Password Requirements */
        .password-requirements {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 6px;
            margin-top: 8px;
        }
        
        .requirement {
            display: flex;
            align-items: center;
            color: var(--text-light);
            font-size: 12px;
        }
        
        .requirement-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 6px;
            background-color: #cbd5e0;
        }
        
        .requirement.met .requirement-indicator {
            background-color: #48bb78;
        }
        
        /* PIN Input */
        .pin-container {
            margin-top: 8px;
        }
        
        .pin-box {
            display: flex;
            gap: 8px;
            justify-content: space-between;
        }
        
        .pin-box input {
            width: 40px;
            height: 40px;
            text-align: center;
            font-size: 16px;
            border: 1px solid var(--border);
            border-radius: 6px;
            -webkit-appearance: none;
            -moz-appearance: textfield;
        }
        
        .pin-box input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.15);
            outline: none;
        }
        
        /* Error & Success Messages */
        .message {
            padding: 10px 12px;
            border-radius: 6px;
            margin-bottom: 16px;
            font-size: 13px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .error-message {
            background-color: #fff5f5;
            color: var(--error);
        }
        
        .error-message i {
            color: var(--error);
        }
        
        .success-message {
            background-color: #f0fff4;
            color: var(--success);
        }
        
        .success-message i {
            color: var(--success);
        }
        
        /* Field Error */
        .field-error {
            color: var(--error);
            font-size: 12px;
            margin-top: 4px;
            display: flex;
            align-items: center;
            gap: 4px;
        }
        
        input.input-error {
            border-color: var(--error);
        }
        
        input.input-error:focus {
            box-shadow: 0 0 0 3px rgba(229, 62, 62, 0.15);
        }
        
        /* Button */
        .btn {
            display: block;
            width: 100%;
            padding: 0;
            height: 44px;
            background-color: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        
        .btn:hover {
            background-color: var(--primary-dark);
            transform: translateY(-1px);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        /* Form Footer */
        .form-footer {
            text-align: center;
            margin-top: 20px;
            color: var(--text-light);
            font-size: 13px;
        }
        
        .form-footer a {
            color: var(--primary);
            text-decoration: none;
            font-weight: 500;
        }
        
        .form-footer a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
<div class="container">
    <div class="form-container">
        <h2><i class="fas fa-user-plus"></i> Create Account</h2>
        
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

        <form method="POST">
            <!-- Full Name -->
            <div class="form-group">
                <label for="full_name">Full Name</label>
                <div class="input-group">
                    <i class="fas fa-user input-icon"></i>
                    <input type="text" id="full_name" name="full_name" 
                           placeholder="Enter your full name" 
                           value="<?= htmlspecialchars($form_data['full_name']) ?>" required>
                </div>
            </div>
            
            <!-- Username -->
            <div class="form-group">
                <label for="username">Username</label>
                <div class="input-group">
                    <i class="fas fa-at input-icon"></i>
                    <input type="text" id="username" name="username" 
                           placeholder="Choose a username" 
                           value="<?= htmlspecialchars($form_data['username']) ?>" 
                           class="<?= $username_error ? 'input-error' : '' ?>" required>
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
                <label for="email">Email</label>
                <div class="input-group">
                    <i class="fas fa-envelope input-icon"></i>
                    <input type="email" id="email" name="email" 
                           placeholder="Enter your email address" 
                           value="<?= htmlspecialchars($form_data['email']) ?>" 
                           class="<?= $email_error ? 'input-error' : '' ?>" required>
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
                    <i class="fas fa-lock input-icon"></i>
                    <input type="password" id="password" name="password" 
                           placeholder="Create a strong password" required>
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
                <label for="pin">6-digit PIN</label>
                <div class="pin-container">
                    <div class="pin-box">
                        <?php for ($i = 0; $i < 6; $i++): ?>
                            <input type="password" name="pin[]" maxlength="1" pattern="\d" inputmode="numeric" required>
                        <?php endfor; ?>
                    </div>
                </div>
            </div>
            
            <!-- Submit Button -->
            <button type="submit" class="btn">
                <i class="fas fa-user-plus"></i> Register
            </button>
        </form>
        
        <div class="form-footer">
            <p>Already have an account? <a href="scan.php">Sign In</a></p>
        </div>
    </div>
</div>

<script>
    document.addEventListener("DOMContentLoaded", () => {
        // PIN input handling
        const pinInputs = document.querySelectorAll(".pin-box input");
        
        // Focus the first PIN input on page load
        setTimeout(() => {
            if (document.activeElement.tagName !== 'INPUT') {
                document.getElementById('full_name').focus();
            }
        }, 500);
        
        // Focus handling for PIN inputs
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
                
                // Allow left/right arrow navigation between fields
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
    });
</script>
</body>
</html>