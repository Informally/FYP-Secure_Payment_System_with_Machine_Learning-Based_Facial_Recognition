<?php
session_start();
require_once '../customer_side/config.php';

// Check if user is authenticated from facial recognition
if (!isset($_SESSION['user_id']) && !isset($_SESSION['authenticatedUserId'])) {
    header("Location: facial_payment.php");
    exit();
}

// FIXED: Get both user IDs like the customer dashboard does
$user_string_id = $_SESSION['user_id'] ?? $_SESSION['authenticatedUserId'] ?? null;
$user_db_id = $_SESSION['user_db_id'] ?? null;

if (!$user_string_id) {
    header("Location: facial_payment.php");
    exit();
}

// FIXED: If we don't have user_db_id in session, get it from database
if (!$user_db_id) {
    $conn = dbConnect();
    $user_check_stmt = $conn->prepare("SELECT id, full_name FROM users WHERE user_id = ?");
    $user_check_stmt->bind_param("s", $user_string_id);
    $user_check_stmt->execute();
    $user_result = $user_check_stmt->get_result();

    if ($user_result->num_rows === 0) {
        // User doesn't exist - clear session and redirect
        unset($_SESSION['user_id']);
        unset($_SESSION['authenticatedUserId']);
        $_SESSION['payment_error'] = "User account not found. Please try again.";
        header("Location: payment_failed.php");
        exit();
    }

    $user_data = $user_result->fetch_assoc();
    $user_db_id = $user_data['id']; // This is the integer ID we need
    $user_check_stmt->close();
    $conn->close();
} else {
    // If we have user_db_id, just get the name
    $conn = dbConnect();
    $user_check_stmt = $conn->prepare("SELECT full_name FROM users WHERE id = ?");
    $user_check_stmt->bind_param("i", $user_db_id);
    $user_check_stmt->execute();
    $user_result = $user_check_stmt->get_result();
    
    if ($user_result->num_rows === 0) {
        unset($_SESSION['user_id']);
        unset($_SESSION['authenticatedUserId']);
        unset($_SESSION['user_db_id']);
        $_SESSION['payment_error'] = "User account not found. Please try again.";
        header("Location: payment_failed.php");
        exit();
    }
    
    $user_data = $user_result->fetch_assoc();
    $user_check_stmt->close();
    $conn->close();
}

$error = "";
$success = false;

if ($_SERVER["REQUEST_METHOD"] === "POST") {
    $conn = dbConnect();
    $pin_input = implode('', $_POST['pin'] ?? []);
    
    if (strlen($pin_input) === 6) {
        // Get user's PIN from users table using integer ID
        $stmt = $conn->prepare("SELECT pin_code FROM users WHERE id = ?");
        $stmt->bind_param("i", $user_db_id);
        $stmt->execute();
        $stmt->bind_result($encrypted_pin);
        $stmt->fetch();
        $stmt->close();

        // Get balance from wallets table using integer ID (like customer dashboard)
        $balance_stmt = $conn->prepare("SELECT balance FROM wallets WHERE user_id = ?");
        $balance_stmt->bind_param("i", $user_db_id);
        $balance_stmt->execute();
        $balance_stmt->bind_result($current_balance);
        $balance_stmt->fetch();
        $balance_stmt->close();

        // If no wallet exists, create one with 0 balance
        if ($current_balance === null) {
            $create_wallet_stmt = $conn->prepare("INSERT INTO wallets (user_id, balance) VALUES (?, 0.00)");
            $create_wallet_stmt->bind_param("i", $user_db_id);
            
            if (!$create_wallet_stmt->execute()) {
                $create_wallet_stmt->close();
                $conn->close();
                $_SESSION['payment_error'] = "Failed to initialize wallet. Please contact support.";
                header("Location: payment_failed.php");
                exit();
            }
            $create_wallet_stmt->close();
            $current_balance = 0.00;
        }
        
        if (aes_decrypt($encrypted_pin) === $pin_input) {
            // PIN is correct, process payment
            
            // Calculate payment amount
            $payment_amount = 0;
            if (isset($_SESSION['kioskCart'])) {
                $cart = $_SESSION['kioskCart'];
                $subtotal = 0;
                foreach ($cart as $item) {
                    $subtotal += $item['price'] * $item['quantity'];
                }
                $tax = $subtotal * 0.06;
                $payment_amount = $subtotal + $tax;
            }
            
            if ($payment_amount > 0) {
                // Check if user has sufficient balance
                if ($current_balance >= $payment_amount) {
                    // Deduct amount from balance
                    $new_balance = $current_balance - $payment_amount;
                    
                    // Update wallet using integer ID (with proper encryption like customer app)
                    $encrypted_balance = aes_encrypt($new_balance);
                    $update_stmt = $conn->prepare("UPDATE wallets SET balance = ?, balance_encrypted = ? WHERE user_id = ?");
                    $update_stmt->bind_param("dsi", $new_balance, $encrypted_balance, $user_db_id);
                    
                    if ($update_stmt->execute()) {
                        // Insert successful transaction record using integer ID (like customer dashboard)
                        $order_id = 'ORDER-' . date('YmdHis') . '-' . $user_db_id;
                        $transaction_stmt = $conn->prepare("
                            INSERT INTO transactions (
                                user_id, amount, status, payment_method,
                                timestamp, order_id, merchant_id, merchant_name, transaction_type
                            ) VALUES (?, ?, 'success', 'facial', NOW(), ?, ?, ?, 'payment')
                        ");
                        $merchant_id = 'KIOSK-001';
                        $merchant_name = 'Prototype Kiosk Store';
                        // FIXED: Store as negative amount to indicate outgoing payment
                        $negative_amount = -$payment_amount;
                        $transaction_stmt->bind_param("idsss",
                            $user_db_id, $negative_amount, $order_id,
                            $merchant_id, $merchant_name
                        );
                        $transaction_stmt->execute();
                        $transaction_stmt->close();
                        
                        // Get the auto-generated transaction ID for session storage
                        $transaction_id = $conn->insert_id;
                        
                        // Store payment details in session for success page
                        $_SESSION['payment_successful'] = true;
                        $_SESSION['transaction_amount'] = $payment_amount;
                        $_SESSION['new_balance'] = $new_balance;
                        $_SESSION['transaction_id'] = $transaction_id;
                       
                        // Clear cart and user session for kiosk security
                        unset($_SESSION['kioskCart']);
                        // FIXED: Clear all user authentication data after successful payment
                        unset($_SESSION['authenticatedUserId']);
                        unset($_SESSION['user_id']);
                        unset($_SESSION['user_db_id']);
                        unset($_SESSION['username']);
                       
                        $update_stmt->close();
                        header("Location: payment_success.php");
                        exit();
                    } else {
                        // Log failed transaction - wallet update failed
                        $order_id = 'ORDER-' . date('YmdHis') . '-' . $user_db_id;
                        $failed_stmt = $conn->prepare("
                            INSERT INTO transactions (
                                user_id, amount, status, payment_method,
                                timestamp, order_id, merchant_id, merchant_name, transaction_type
                            ) VALUES (?, ?, 'failed', 'facial', NOW(), ?, ?, ?, 'payment')
                        ");
                        $merchant_id = 'KIOSK-001';
                        $merchant_name = 'Prototype Kiosk Store';
                        $negative_amount = -$payment_amount;
                        $failed_stmt->bind_param("idsss",
                            $user_db_id, $negative_amount, $order_id,
                            $merchant_id, $merchant_name
                        );
                        $failed_stmt->execute();
                        $failed_stmt->close();
                        
                        $error = "Payment processing failed. Please try again.";
                    }
                    $update_stmt->close();
                } else {
                    // Log failed transaction - insufficient balance
                    $order_id = 'ORDER-' . date('YmdHis') . '-' . $user_db_id;
                    $failed_stmt = $conn->prepare("
                        INSERT INTO transactions (
                            user_id, amount, status, payment_method,
                            timestamp, order_id, merchant_id, merchant_name, transaction_type
                        ) VALUES (?, ?, 'failed', 'facial', NOW(), ?, ?, ?, 'payment')
                    ");
                    $merchant_id = 'KIOSK-001';
                    $merchant_name = 'Prototype Kiosk Store';
                    $negative_amount = -$payment_amount;
                    $failed_stmt->bind_param("idsss",
                        $user_db_id, $negative_amount, $order_id,
                        $merchant_id, $merchant_name
                    );
                    $failed_stmt->execute();
                    $failed_stmt->close();
                    
                    $error = "Insufficient balance. Required: RM " . number_format($payment_amount, 2) . ", Available: RM " . number_format($current_balance, 2);
                }
            } else {
                $error = "Invalid payment amount.";
            }
        } else {
            // Log failed transaction - invalid PIN
            $payment_amount = 0;
            if (isset($_SESSION['kioskCart'])) {
                $cart = $_SESSION['kioskCart'];
                $subtotal = 0;
                foreach ($cart as $item) {
                    $subtotal += $item['price'] * $item['quantity'];
                }
                $tax = $subtotal * 0.06;
                $payment_amount = $subtotal + $tax;
            }
            
            if ($payment_amount > 0) {
                $order_id = 'ORDER-' . date('YmdHis') . '-' . $user_db_id;
                $failed_stmt = $conn->prepare("
                    INSERT INTO transactions (
                        user_id, amount, status, payment_method,
                        timestamp, order_id, merchant_id, merchant_name, transaction_type
                    ) VALUES (?, ?, 'failed', 'facial', NOW(), ?, ?, ?, 'payment')
                ");
                $merchant_id = 'KIOSK-001';
                $merchant_name = 'Prototype Kiosk Store';
                $negative_amount = -$payment_amount;
                $failed_stmt->bind_param("idsss",
                    $user_db_id, $negative_amount, $order_id,
                    $merchant_id, $merchant_name
                );
                $failed_stmt->execute();
                $failed_stmt->close();
            }
            
            $error = "Invalid PIN. Please try again.";
        }
    } else {
        $error = "Please enter a complete 6-digit PIN.";
    }
    $conn->close();
}

// Get cart data for display
$cart = $_SESSION['kioskCart'] ?? [];
$payment_amount = 0;
if (!empty($cart)) {
    $subtotal = 0;
    foreach ($cart as $item) {
        $subtotal += $item['price'] * $item['quantity'];
    }
    $tax = $subtotal * 0.06;
    $payment_amount = $subtotal + $tax;
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enter PIN - Prototype Kiosk</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Arial', sans-serif;
            background: #f8f8f8;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .kiosk-header {
            background: linear-gradient(45deg, #da291c, #ffc72c);
            padding: 15px 0;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        .kiosk-header h1 {
            color: white;
            font-size: 2.5rem;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .pin-container {
            max-width: 800px;
            margin: 20px auto;
            padding: 0 20px;
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .pin-card {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            border: 4px solid #27ae60;
            text-align: center;
            width: 100%;
            max-width: 500px;
        }

        .pin-header {
            margin-bottom: 30px;
        }

        .pin-header i {
            font-size: 64px;
            color: #27ae60;
            margin-bottom: 20px;
        }

        .pin-header h2 {
            font-size: 28px;
            color: #2c3e50;
            margin-bottom: 10px;
        }

        .pin-header p {
            color: #7f8c8d;
            font-size: 16px;
            line-height: 1.5;
        }

        .user-info {
            background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
            border: 2px solid #4caf50;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
        }

        .user-info h3 {
            color: #2e7d32;
            margin-bottom: 10px;
            font-size: 20px;
        }

        .user-info p {
            color: #388e3c;
            font-size: 16px;
        }

        .order-summary {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            border: 2px solid #dee2e6;
        }

        .order-summary h4 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 18px;
        }

        .order-total {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 24px;
            font-weight: bold;
            color: #da291c;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 2px solid #dee2e6;
        }

        .pin-display {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 30px;
        }

        .pin-dot {
            width: 20px;
            height: 20px;
            border: 3px solid #bdc3c7;
            border-radius: 50%;
            transition: all 0.3s ease;
        }

        .pin-dot.filled {
            background: #27ae60;
            border-color: #27ae60;
            transform: scale(1.2);
        }

        .pin-keypad {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 30px;
            max-width: 300px;
            margin-left: auto;
            margin-right: auto;
        }

        .pin-key {
            width: 80px;
            height: 80px;
            border: 3px solid #ecf0f1;
            border-radius: 50%;
            background: #f8f9fa;
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .pin-key:hover {
            background: #3498db;
            color: white;
            border-color: #2980b9;
            transform: scale(1.1);
        }

        .pin-key:active {
            transform: scale(0.95);
        }

        .pin-key.clear {
            background: #e74c3c;
            color: white;
            border-color: #c0392b;
        }

        .pin-key.clear:hover {
            background: #c0392b;
        }

        .pin-box {
            display: none; /* Hide the original input boxes */
        }

        .action-buttons {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 30px;
        }

        .btn {
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 10px;
        }

        .btn-success {
            background: linear-gradient(45deg, #27ae60, #2ecc71);
            color: white;
        }

        .btn-success:hover {
            background: linear-gradient(45deg, #2ecc71, #27ae60);
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(39, 174, 96, 0.3);
        }

        .btn-secondary {
            background: #95a5a6;
            color: white;
        }

        .btn-secondary:hover {
            background: #7f8c8d;
            transform: translateY(-2px);
        }

        .error-message {
            background: #ffebee;
            color: #c62828;
            border: 2px solid #f44336;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            font-size: 16px;
            font-weight: bold;
            text-align: center;
        }

        .success-message {
            background: #e8f5e9;
            color: #2e7d32;
            border: 2px solid #4caf50;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            font-size: 16px;
            font-weight: bold;
            text-align: center;
        }

        @media (max-width: 768px) {
            .pin-container {
                padding: 20px;
            }

            .pin-card {
                padding: 30px 20px;
            }

            .pin-box input {
                width: 50px;
                height: 50px;
                font-size: 20px;
            }

            .kiosk-header h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="kiosk-header">
        <h1><i class="fas fa-lock"></i> Enter PIN - Prototype Kiosk</h1>
    </div>

    <div class="pin-container">
        <div class="pin-card">
            <div class="pin-header">
                <i class="fas fa-shield-alt"></i>
                <h2>Secure Payment</h2>
                <p>Please enter your 6-digit PIN to complete the payment</p>
            </div>

            <div class="user-info">
                <h3><i class="fas fa-user-check"></i> Identity Verified</h3>
                <p>User: <?php echo htmlspecialchars($user_data['full_name'] ?? 'User'); ?> (ID: <?php echo htmlspecialchars($user_db_id); ?>)</p>
            </div>

            <div class="order-summary">
                <h4><i class="fas fa-receipt"></i> Payment Amount</h4>
                <div class="order-total">
                    <span>Total:</span>
                    <span>RM <?php echo number_format($payment_amount, 2); ?></span>
                </div>
            </div>

            <?php if (!empty($error)): ?>
                <div class="error-message"><?php echo htmlspecialchars($error); ?></div>
            <?php endif; ?>

            <!-- Visual PIN Display -->
            <div class="pin-display">
                <div class="pin-dot" id="dot1"></div>
                <div class="pin-dot" id="dot2"></div>
                <div class="pin-dot" id="dot3"></div>
                <div class="pin-dot" id="dot4"></div>
                <div class="pin-dot" id="dot5"></div>
                <div class="pin-dot" id="dot6"></div>
            </div>

            <!-- Visual Keypad -->
            <div class="pin-keypad">
                <div class="pin-key" onclick="enterDigit('1')">1</div>
                <div class="pin-key" onclick="enterDigit('2')">2</div>
                <div class="pin-key" onclick="enterDigit('3')">3</div>
                <div class="pin-key" onclick="enterDigit('4')">4</div>
                <div class="pin-key" onclick="enterDigit('5')">5</div>
                <div class="pin-key" onclick="enterDigit('6')">6</div>
                <div class="pin-key" onclick="enterDigit('7')">7</div>
                <div class="pin-key" onclick="enterDigit('8')">8</div>
                <div class="pin-key" onclick="enterDigit('9')">9</div>
                <div class="pin-key clear" onclick="clearPin()">
                    <i class="fas fa-backspace"></i>
                </div>
                <div class="pin-key" onclick="enterDigit('0')">0</div>
                <div class="pin-key clear" onclick="clearAllPin()">
                    <i class="fas fa-times"></i>
                </div>
            </div>

            <form method="POST" id="pinForm">
                <!-- Hidden inputs for PIN -->
                <div class="pin-box">
                    <?php for ($i = 0; $i < 6; $i++): ?>
                        <input type="password" name="pin[]" maxlength="1" pattern="\d" id="pin<?php echo $i; ?>">
                    <?php endfor; ?>
                </div>

                <div class="action-buttons">
                    <button type="submit" class="btn btn-success" id="confirmBtn" disabled>
                        <i class="fas fa-check"></i> Confirm Payment
                    </button>
                    <button type="button" class="btn btn-secondary" onclick="goBack()">
                        <i class="fas fa-arrow-left"></i> Back
                    </button>
                </div>
            </form>
        </div>
    </div>

    <script>
        let enteredPin = '';

        function enterDigit(digit) {
            if (enteredPin.length < 6) {
                enteredPin += digit;
                updatePinDisplay();
                updateHiddenInputs();
                
                // Auto-submit when 6 digits entered
                if (enteredPin.length === 6) {
                    document.getElementById('confirmBtn').disabled = false;
                    setTimeout(() => {
                        document.getElementById('pinForm').submit();
                    }, 500);
                }
            }
        }

        function clearPin() {
            if (enteredPin.length > 0) {
                enteredPin = enteredPin.slice(0, -1);
                updatePinDisplay();
                updateHiddenInputs();
            }
        }

        function clearAllPin() {
            enteredPin = '';
            updatePinDisplay();
            updateHiddenInputs();
        }

        function updatePinDisplay() {
            for (let i = 1; i <= 6; i++) {
                const dot = document.getElementById(`dot${i}`);
                if (i <= enteredPin.length) {
                    dot.classList.add('filled');
                } else {
                    dot.classList.remove('filled');
                }
            }

            // Enable/disable confirm button
            document.getElementById('confirmBtn').disabled = enteredPin.length !== 6;
        }

        function updateHiddenInputs() {
            // Update the hidden form inputs with the entered PIN
            for (let i = 0; i < 6; i++) {
                const input = document.getElementById(`pin${i}`);
                if (i < enteredPin.length) {
                    input.value = enteredPin[i];
                } else {
                    input.value = '';
                }
            }
        }

        function goBack() {
            window.location.href = 'facial_payment.php';
        }

        // Keyboard support
        document.addEventListener('keydown', function(e) {
            if (e.key >= '0' && e.key <= '9') {
                enterDigit(e.key);
            } else if (e.key === 'Backspace') {
                clearPin();
            } else if (e.key === 'Enter') {
                if (enteredPin.length === 6) {
                    document.getElementById('pinForm').submit();
                }
            } else if (e.key === 'Escape') {
                clearAllPin();
            }
        });

        // Prevent form submission if PIN is incomplete
        document.getElementById('pinForm').addEventListener('submit', function(e) {
            if (enteredPin.length !== 6) {
                e.preventDefault();
                alert('Please enter a complete 6-digit PIN');
                return false;
            }
        });
    </script>
</body>
</html>