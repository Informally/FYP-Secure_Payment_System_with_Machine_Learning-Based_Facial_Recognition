<?php
// view_balance.php - Sensitive Authentication Required for Balance View (MySQLi version - PIN only)
require_once 'config.php';
require_once 'encrypt.php';

// Check basic authentication first
if (!SessionManager::start() || !SessionManager::requireAuthLevel('basic')) {
    header("Location: login.php");
    exit();
}

$user_id = $_SESSION['authenticatedUserId'];
$user_db_id = $_SESSION['user_db_id'];

// Check if sensitive authentication is required
$requires_auth = !SessionManager::requireAuthLevel('sensitive');
$auth_error = "";
$auth_success = false;

// Handle sensitive authentication (PIN only)
if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST['action']) && $_POST['action'] === 'authenticate') {
    // CSRF validation
    if (!validateCSRFToken($_POST['csrf_token'] ?? '')) {
        $auth_error = "Invalid request. Please try again.";
    } else {
        $entered_pin = implode('', $_POST['pin'] ?? []);
        
        if (empty($entered_pin) || !preg_match('/^\d{6}$/', $entered_pin)) {
            $auth_error = "Please enter a valid 6-digit PIN.";
        } else {
            try {
                $conn = dbConnect(); // Use MySQLi connection
                
                // Verify PIN (MySQLi style)
                $stmt = $conn->prepare("SELECT pin_code FROM users WHERE id = ?");
                $stmt->bind_param("i", $user_db_id);
                $stmt->execute();
                $result = $stmt->get_result();
                $row = $result->fetch_assoc();
                $stored_pin = $row['pin_code'] ?? null;
                
                if ($stored_pin && aes_decrypt($stored_pin) === $entered_pin) {
                    // Upgrade to sensitive authentication
                    SessionManager::setAuthLevel('sensitive', $user_id);
                    $auth_success = true;
                    $requires_auth = false;
                    
                    SecurityUtils::logSecurityEvent($user_id, 'sensitive_auth_success', 'success', [
                        'method' => 'pin',
                        'action' => 'view_balance'
                    ]);
                } else {
                    $auth_error = "Invalid PIN. Please try again.";
                    SecurityUtils::logSecurityEvent($user_id, 'sensitive_auth_failed', 'failure', [
                        'method' => 'pin',
                        'action' => 'view_balance'
                    ]);
                }
                
                $stmt->close();
                $conn->close();
                
            } catch (Exception $e) {
                error_log("Sensitive auth error: " . $e->getMessage());
                $auth_error = "Authentication failed. Please try again.";
            }
        }
    }
}

// Get wallet data if authenticated
$wallet_data = null;
$transaction_summary = null;

if (!$requires_auth) {
    try {
        $conn = dbConnect(); // Use MySQLi connection
        
        // Get wallet information (MySQLi style)
        $stmt = $conn->prepare("
            SELECT balance_encrypted, daily_limit_encrypted, monthly_limit_encrypted,
                   daily_spent_encrypted, monthly_spent_encrypted, created_at, updated_at
            FROM wallets 
            WHERE user_id = ?
        ");
        $stmt->bind_param("i", $user_db_id);
        $stmt->execute();
        $result = $stmt->get_result();
        $wallet = $result->fetch_assoc();
        
        if ($wallet) {
            $wallet_data = [
                'balance' => aes_decrypt($wallet['balance_encrypted']),
                'daily_limit' => aes_decrypt($wallet['daily_limit_encrypted']),
                'monthly_limit' => aes_decrypt($wallet['monthly_limit_encrypted']),
                'daily_spent' => $wallet['daily_spent_encrypted'] ? aes_decrypt($wallet['daily_spent_encrypted']) : '0.00',
                'monthly_spent' => $wallet['monthly_spent_encrypted'] ? aes_decrypt($wallet['monthly_spent_encrypted']) : '0.00',
                'created_at' => $wallet['created_at'],
                'updated_at' => $wallet['updated_at']
            ];
            
            // Calculate remaining limits
            $wallet_data['daily_remaining'] = max(0, floatval($wallet_data['daily_limit']) - floatval($wallet_data['daily_spent']));
            $wallet_data['monthly_remaining'] = max(0, floatval($wallet_data['monthly_limit']) - floatval($wallet_data['monthly_spent']));
        }
        
        $stmt->close();
        
        // Get transaction summary for current month (MySQLi style)
        // Note: Simplified query since MySQLi doesn't support AES_DECRYPT in the same way
        $stmt = $conn->prepare("
            SELECT 
                transaction_type,
                COUNT(*) as count
            FROM transactions 
            WHERE user_id = ? 
            AND MONTH(created_at) = MONTH(NOW()) 
            AND YEAR(created_at) = YEAR(NOW())
            AND status = 'success'
            GROUP BY transaction_type
        ");
        $stmt->bind_param("i", $user_db_id);
        $stmt->execute();
        $result = $stmt->get_result();
        $transaction_summary = [];
        
        while ($row = $result->fetch_assoc()) {
            // Get individual transaction amounts for this type
            $amount_stmt = $conn->prepare("
                SELECT amount_encrypted, amount
                FROM transactions 
                WHERE user_id = ? 
                AND transaction_type = ?
                AND MONTH(created_at) = MONTH(NOW()) 
                AND YEAR(created_at) = YEAR(NOW())
                AND status = 'success'
            ");
            $amount_stmt->bind_param("is", $user_db_id, $row['transaction_type']);
            $amount_stmt->execute();
            $amount_result = $amount_stmt->get_result();
            
            $total_amount = 0;
            while ($amount_row = $amount_result->fetch_assoc()) {
                if ($amount_row['amount_encrypted']) {
                    $total_amount += floatval(aes_decrypt($amount_row['amount_encrypted']));
                } else {
                    $total_amount += floatval($amount_row['amount']);
                }
            }
            
            $transaction_summary[] = [
                'transaction_type' => $row['transaction_type'],
                'count' => $row['count'],
                'total_amount' => $total_amount
            ];
            
            $amount_stmt->close();
        }
        
        $stmt->close();
        $conn->close();
        
        SecurityUtils::logSecurityEvent($user_id, 'balance_viewed', 'success');
        
    } catch (Exception $e) {
        error_log("Balance view error: " . $e->getMessage());
        $auth_error = "Unable to load wallet data.";
    }
}

$csrf_token = generateCSRFToken();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wallet Balance - Secure FacePay</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --primary: #3498db;
            --primary-light: #e3f2fd;
            --primary-dark: #2980b9;
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
            background-color: var(--bg-light);
            color: var(--text-dark);
            line-height: 1.6;
        }
        
        /* Header */
        .header {
            background: linear-gradient(135deg, var(--primary), #667eea);
            color: white;
            padding: 20px 0;
            box-shadow: 0 2px 10px var(--shadow);
        }
        
        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .back-btn {
            color: white;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 8px;
            transition: background-color 0.3s;
        }
        
        .back-btn:hover {
            background-color: rgba(255,255,255,0.1);
            text-decoration: none;
            color: white;
        }
        
        .page-title {
            font-size: 24px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .security-badge {
            background: rgba(255,255,255,0.2);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Main Content */
        .main-content {
            max-width: 800px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        
        /* Authentication Form */
        .auth-container {
            background: white;
            border-radius: 16px;
            padding: 40px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            text-align: center;
            max-width: 500px;
            margin: 0 auto;
        }
        
        .auth-icon {
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, var(--primary), #667eea);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 25px;
            font-size: 32px;
            color: white;
        }
        
        .auth-title {
            font-size: 24px;
            font-weight: 700;
            color: var(--text-dark);
            margin-bottom: 10px;
        }
        
        .auth-description {
            color: var(--text-light);
            margin-bottom: 30px;
            line-height: 1.6;
        }
        
        /* PIN Input */
        .pin-container {
            display: flex;
            gap: 12px;
            justify-content: center;
            margin: 30px 0;
        }
        
        .pin-input {
            width: 50px;
            height: 50px;
            text-align: center;
            font-size: 18px;
            font-weight: 600;
            border: 2px solid var(--border);
            border-radius: 10px;
            background-color: #fafafa;
            transition: all 0.3s ease;
        }
        
        .pin-input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
            outline: none;
            background-color: white;
        }
        
        .pin-input.filled {
            border-color: var(--success);
            background-color: var(--primary-light);
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
        
        .success-message {
            background-color: #f0fff4;
            color: var(--success);
            border: 1px solid #c6f6d5;
        }
        
        .error-message {
            background-color: #fff5f5;
            color: var(--error);
            border: 1px solid #fed7d7;
        }
        
        /* Buttons */
        .btn {
            padding: 15px 30px;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            text-decoration: none;
        }
        
        .btn-primary {
            background: var(--primary);
            color: white;
            width: 100%;
            justify-content: center;
        }
        
        .btn-primary:hover {
            background: var(--primary-dark);
            transform: translateY(-2px);
        }
        
        .btn-primary:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        /* PIN Helper Text */
        .pin-helper {
            margin-top: 15px;
            font-size: 13px;
            color: var(--text-light);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        
        /* Wallet Display */
        .wallet-container {
            display: grid;
            gap: 25px;
        }
        
        .balance-card {
            background: linear-gradient(135deg, var(--primary), #667eea);
            color: white;
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            box-shadow: 0 8px 30px rgba(52, 152, 219, 0.3);
        }
        
        .balance-label {
            font-size: 16px;
            margin-bottom: 10px;
            opacity: 0.9;
        }
        
        .balance-amount {
            font-size: 48px;
            font-weight: 700;
            margin-bottom: 15px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .balance-updated {
            font-size: 12px;
            opacity: 0.8;
        }
        
        .limits-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 25px;
        }
        
        .limit-card {
            background: white;
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        }
        
        .limit-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 20px;
        }
        
        .limit-icon {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            color: white;
        }
        
        .icon-daily { background: var(--warning); }
        .icon-monthly { background: var(--success); }
        .icon-primary { background: var(--primary); }
        
        .limit-title {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-dark);
        }
        
        .limit-details {
            display: grid;
            gap: 15px;
        }
        
        .limit-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid var(--border);
        }
        
        .limit-item:last-child {
            border-bottom: none;
        }
        
        .limit-label {
            color: var(--text-light);
            font-size: 14px;
        }
        
        .limit-value {
            font-weight: 600;
            color: var(--text-dark);
        }
        
        .limit-remaining {
            color: var(--success);
        }
        
        .limit-spent {
            color: var(--warning);
        }
        
        /* Progress Bars */
        .progress-bar {
            width: 100%;
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }
        
        .progress-fill {
            height: 100%;
            background: var(--primary);
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        
        .progress-warning {
            background: var(--warning);
        }
        
        .progress-danger {
            background: var(--error);
        }
        
        /* Transaction Summary */
        .summary-card {
            background: white;
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            margin-top: 25px;
        }
        
        .summary-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--border);
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        
        .summary-item {
            text-align: center;
            padding: 20px;
            background: var(--bg-light);
            border-radius: 12px;
            transition: transform 0.3s ease;
        }
        
        .summary-item:hover {
            transform: translateY(-2px);
        }
        
        .summary-amount {
            font-size: 24px;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 5px;
        }
        
        .summary-label {
            font-size: 12px;
            color: var(--text-light);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .summary-count {
            font-size: 14px;
            color: var(--text-medium);
            margin-top: 5px;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .main-content {
                padding: 20px 15px;
            }
            
            .auth-container {
                padding: 30px 20px;
            }
            
            .pin-container {
                gap: 8px;
            }
            
            .pin-input {
                width: 45px;
                height: 45px;
                font-size: 16px;
            }
            
            .balance-amount {
                font-size: 36px;
            }
            
            .limits-grid {
                grid-template-columns: 1fr;
            }
            
            .summary-grid {
                grid-template-columns: 1fr;
            }
        }
        
        @media (max-width: 480px) {
            .pin-container {
                gap: 6px;
            }
            
            .pin-input {
                width: 40px;
                height: 40px;
                font-size: 14px;
            }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="header-content">
            <a href="dashboard.php" class="back-btn">
                <i class="fas fa-arrow-left"></i>
                Back to Dashboard
            </a>
            <div class="page-title">
                <i class="fas fa-wallet"></i>
                Wallet Balance
            </div>
            <div class="security-badge">
                <i class="fas fa-shield-check"></i>
                PIN Required
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="main-content">
        <?php if ($requires_auth): ?>
            <!-- PIN Authentication Required -->
            <div class="auth-container">
                <div class="auth-icon">
                    <i class="fas fa-hashtag"></i>
                </div>
                
                <h2 class="auth-title">Enter Your PIN</h2>
                <p class="auth-description">
                    To view your wallet balance and financial details, please enter your 6-digit PIN for secure access.
                </p>
                
                <?php if ($auth_error): ?>
                    <div class="message error-message">
                        <i class="fas fa-exclamation-circle"></i>
                        <?= htmlspecialchars($auth_error) ?>
                    </div>
                <?php endif; ?>
                
                <form method="POST" id="authForm">
                    <input type="hidden" name="csrf_token" value="<?= $csrf_token ?>">
                    <input type="hidden" name="action" value="authenticate">
                    
                    <div class="pin-container">
                        <?php for ($i = 0; $i < 6; $i++): ?>
                            <input type="password" name="pin[]" class="pin-input" 
                                   maxlength="1" pattern="\d" inputmode="numeric" required
                                   autocomplete="off">
                        <?php endfor; ?>
                    </div>
                    
                    <div class="pin-helper">
                        <i class="fas fa-info-circle"></i>
                        Enter your 6-digit PIN to access your wallet
                    </div>
                    
                    <button type="submit" class="btn btn-primary">
                        <i class="fas fa-unlock"></i>
                        View My Balance
                    </button>
                </form>
            </div>
            
        <?php else: ?>
            <!-- Wallet Information Display -->
            <?php if ($auth_success): ?>
                <div class="message success-message">
                    <i class="fas fa-check-circle"></i>
                    PIN verified successfully! Your wallet information is now visible.
                </div>
            <?php endif; ?>
            
            <div class="wallet-container">
                <!-- Main Balance Card -->
                <div class="balance-card">
                    <div class="balance-label">Current Balance</div>
                    <div class="balance-amount">RM <?= number_format($wallet_data['balance'], 2) ?></div>
                    <div class="balance-updated">
                        Last updated: <?= date('M j, Y \a\t g:i A', strtotime($wallet_data['updated_at'])) ?>
                    </div>
                </div>
                
                <!-- Spending Limits -->
                <div class="limits-grid">
                    <!-- Daily Limit -->
                    <div class="limit-card">
                        <div class="limit-header">
                            <div class="limit-icon icon-daily">
                                <i class="fas fa-calendar-day"></i>
                            </div>
                            <div class="limit-title">Daily Spending</div>
                        </div>
                        
                        <div class="limit-details">
                            <div class="limit-item">
                                <span class="limit-label">Daily Limit</span>
                                <span class="limit-value">RM <?= number_format($wallet_data['daily_limit'], 2) ?></span>
                            </div>
                            <div class="limit-item">
                                <span class="limit-label">Spent Today</span>
                                <span class="limit-value limit-spent">RM <?= number_format($wallet_data['daily_spent'], 2) ?></span>
                            </div>
                            <div class="limit-item">
                                <span class="limit-label">Remaining</span>
                                <span class="limit-value limit-remaining">RM <?= number_format($wallet_data['daily_remaining'], 2) ?></span>
                            </div>
                        </div>
                        
                        <?php 
                        $daily_percentage = ($wallet_data['daily_limit'] > 0) ? 
                            ($wallet_data['daily_spent'] / $wallet_data['daily_limit']) * 100 : 0;
                        $daily_progress_class = $daily_percentage > 80 ? 'progress-danger' : 
                                              ($daily_percentage > 60 ? 'progress-warning' : '');
                        ?>
                        <div class="progress-bar">
                            <div class="progress-fill <?= $daily_progress_class ?>" 
                                 style="width: <?= min($daily_percentage, 100) ?>%"></div>
                        </div>
                    </div>
                    
                    <!-- Monthly Limit -->
                    <div class="limit-card">
                        <div class="limit-header">
                            <div class="limit-icon icon-monthly">
                                <i class="fas fa-calendar-alt"></i>
                            </div>
                            <div class="limit-title">Monthly Spending</div>
                        </div>
                        
                        <div class="limit-details">
                            <div class="limit-item">
                                <span class="limit-label">Monthly Limit</span>
                                <span class="limit-value">RM <?= number_format($wallet_data['monthly_limit'], 2) ?></span>
                            </div>
                            <div class="limit-item">
                                <span class="limit-label">Spent This Month</span>
                                <span class="limit-value limit-spent">RM <?= number_format($wallet_data['monthly_spent'], 2) ?></span>
                            </div>
                            <div class="limit-item">
                                <span class="limit-label">Remaining</span>
                                <span class="limit-value limit-remaining">RM <?= number_format($wallet_data['monthly_remaining'], 2) ?></span>
                            </div>
                        </div>
                        
                        <?php 
                        $monthly_percentage = ($wallet_data['monthly_limit'] > 0) ? 
                            ($wallet_data['monthly_spent'] / $wallet_data['monthly_limit']) * 100 : 0;
                        $monthly_progress_class = $monthly_percentage > 80 ? 'progress-danger' : 
                                                ($monthly_percentage > 60 ? 'progress-warning' : '');
                        ?>
                        <div class="progress-bar">
                            <div class="progress-fill <?= $monthly_progress_class ?>" 
                                 style="width: <?= min($monthly_percentage, 100) ?>%"></div>
                        </div>
                    </div>
                </div>
                
                <!-- Transaction Summary -->
                <?php if ($transaction_summary && count($transaction_summary) > 0): ?>
                    <div class="summary-card">
                        <div class="summary-header">
                            <div class="limit-icon icon-primary">
                                <i class="fas fa-chart-pie"></i>
                            </div>
                            <div class="limit-title">This Month's Summary</div>
                        </div>
                        
                        <div class="summary-grid">
                            <?php foreach ($transaction_summary as $summary): ?>
                                <div class="summary-item">
                                    <div class="summary-amount">
                                        RM <?= number_format($summary['total_amount'], 2) ?>
                                    </div>
                                    <div class="summary-label">
                                        <?= ucwords(str_replace('_', ' ', $summary['transaction_type'])) ?>
                                    </div>
                                    <div class="summary-count">
                                        <?= $summary['count'] ?> transaction<?= $summary['count'] != 1 ? 's' : '' ?>
                                    </div>
                                </div>
                            <?php endforeach; ?>
                        </div>
                    </div>
                <?php endif; ?>
                
                <!-- Quick Actions -->
                <div class="summary-card">
                    <div class="summary-header">
                        <div class="limit-icon icon-primary">
                            <i class="fas fa-bolt"></i>
                        </div>
                        <div class="limit-title">Quick Actions</div>
                    </div>
                    
                    <div class="summary-grid">
                        <a href="topup.php" class="summary-item" style="text-decoration: none; color: inherit;">
                            <div class="summary-amount" style="color: var(--success);">
                                <i class="fas fa-plus-circle"></i>
                            </div>
                            <div class="summary-label">Top Up Wallet</div>
                            <div class="summary-count">Add funds to your account</div>
                        </a>
                           
                        <a href="transaction_history.php" class="summary-item" style="text-decoration: none; color: inherit;">
                            <div class="summary-amount" style="color: var(--warning);">
                                <i class="fas fa-history"></i>
                            </div>
                            <div class="summary-label">View History</div>
                            <div class="summary-count">See all transactions</div>
                        </a>
                        
                        <a href="profile.php" class="summary-item" style="text-decoration: none; color: inherit;">
                            <div class="summary-amount" style="color: var(--text-medium);">
                                <i class="fas fa-cog"></i>
                            </div>
                            <div class="summary-label">Settings</div>
                            <div class="summary-count">Manage your account</div>
                        </a>
                    </div>
                </div>
            </div>
        <?php endif; ?>
    </main>

    <script>
        // PIN input handling
        const pinInputs = document.querySelectorAll('.pin-input');
        
        pinInputs.forEach((input, index) => {
            input.addEventListener('input', (e) => {
                // Only allow digits
                e.target.value = e.target.value.replace(/[^0-9]/g, '');
                
                // Add visual feedback
                if (e.target.value) {
                    e.target.classList.add('filled');
                } else {
                    e.target.classList.remove('filled');
                }
                
                // Auto-advance to next input
                if (e.target.value.length === 1 && index < pinInputs.length - 1) {
                    pinInputs[index + 1].focus();
                }
                
                // Auto-submit when all fields are filled
                const allFilled = Array.from(pinInputs).every(input => input.value.length === 1);
                if (allFilled) {
                    setTimeout(() => {
                        document.getElementById('authForm').querySelector('.btn').click();
                    }, 200);
                }
            });
            
            input.addEventListener('keydown', (e) => {
                // Handle backspace
                if (e.key === 'Backspace') {
                    if (e.target.value === '' && index > 0) {
                        pinInputs[index - 1].focus();
                        pinInputs[index - 1].value = '';
                        pinInputs[index - 1].classList.remove('filled');
                        e.preventDefault();
                    } else if (e.target.value) {
                        e.target.classList.remove('filled');
                    }
                }
                
                // Handle arrow keys
                if (e.key === 'ArrowLeft' && index > 0) {
                    pinInputs[index - 1].focus();
                    e.preventDefault();
                }
                
                if (e.key === 'ArrowRight' && index < pinInputs.length - 1) {
                    pinInputs[index + 1].focus();
                    e.preventDefault();
                }
                
                // Prevent non-numeric input
                if (!/[0-9]/.test(e.key) && !['Backspace', 'Delete', 'Tab', 'Enter', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
                    e.preventDefault();
                }
            });
            
            input.addEventListener('paste', (e) => {
                e.preventDefault();
                const pasteData = e.clipboardData.getData('text').replace(/[^0-9]/g, '').slice(0, 6);
                
                // Clear all inputs first
                pinInputs.forEach(input => {
                    input.value = '';
                    input.classList.remove('filled');
                });
                
                // Fill inputs with pasted data
                for (let i = 0; i < Math.min(pasteData.length, pinInputs.length); i++) {
                    pinInputs[i].value = pasteData[i];
                    pinInputs[i].classList.add('filled');
                }
                
                // Focus appropriate input
                if (pasteData.length >= pinInputs.length) {
                    pinInputs[pinInputs.length - 1].focus();
                    // Auto-submit if all fields are filled
                    setTimeout(() => {
                        document.getElementById('authForm').querySelector('.btn').click();
                    }, 200);
                } else if (pasteData.length > 0) {
                    pinInputs[pasteData.length].focus();
                }
            });
            
            // Handle focus
            input.addEventListener('focus', (e) => {
                e.target.select();
            });
        });
        
        // Form submission handling
        if (document.getElementById('authForm')) {
            document.getElementById('authForm').addEventListener('submit', function(e) {
                const submitBtn = this.querySelector('.btn');
                const pinValues = Array.from(pinInputs).map(input => input.value);
                
                // Validate PIN is complete
                if (pinValues.some(val => val === '')) {
                    e.preventDefault();
                    alert('Please enter your complete 6-digit PIN');
                    pinInputs.find(input => input.value === '').focus();
                    return;
                }
                
                // Update button state
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Verifying PIN...';
                
                // Re-enable after 3 seconds to prevent permanent lockout
                setTimeout(() => {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '<i class="fas fa-unlock"></i> View My Balance';
                }, 3000);
            });
        }
        
        // Auto-focus first input on page load
        document.addEventListener('DOMContentLoaded', function() {
            const firstPinInput = document.querySelector('.pin-input');
            if (firstPinInput) {
                firstPinInput.focus();
            }
        });
        
        // Animate progress bars
        document.addEventListener('DOMContentLoaded', function() {
            const progressBars = document.querySelectorAll('.progress-fill');
            progressBars.forEach(bar => {
                const width = bar.style.width;
                bar.style.width = '0%';
                setTimeout(() => {
                    bar.style.width = width;
                }, 500);
            });
        });
        
        // Auto-hide success messages
        const successMessage = document.querySelector('.success-message');
        if (successMessage) {
            setTimeout(() => {
                successMessage.style.opacity = '0';
                successMessage.style.transform = 'translateY(-10px)';
                setTimeout(() => {
                    successMessage.remove();
                }, 300);
            }, 5000);
        }
        
        // Session timeout warning for sensitive data
        let sensitiveSessionTimer;
        function startSensitiveSessionTimer() {
            sensitiveSessionTimer = setTimeout(() => {
                if (confirm('Your secure session will expire soon. Do you want to extend it?')) {
                    // Extend session
                    fetch('store_session.php', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            action: 'extend_session'
                        })
                    });
                    startSensitiveSessionTimer(); // Restart timer
                } else {
                    // Redirect to dashboard
                    window.location.href = 'dashboard.php';
                }
            }, 10 * 60 * 1000); // 10 minutes for sensitive data
        }
        
        // Start sensitive session timer if we're viewing balance
        <?php if (!$requires_auth): ?>
        startSensitiveSessionTimer();
        <?php endif; ?>
        
        // Add shake animation for wrong PIN
        <?php if ($auth_error && $_SERVER["REQUEST_METHOD"] == "POST"): ?>
        document.addEventListener('DOMContentLoaded', function() {
            const pinContainer = document.querySelector('.pin-container');
            if (pinContainer) {
                pinContainer.style.animation = 'shake 0.5s ease-in-out';
                setTimeout(() => {
                    pinContainer.style.animation = '';
                    // Clear all PIN inputs and focus first one
                    pinInputs.forEach(input => {
                        input.value = '';
                        input.classList.remove('filled');
                    });
                    pinInputs[0].focus();
                }, 500);
            }
        });
        <?php endif; ?>
    </script>
    
    <style>
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
            20%, 40%, 60%, 80% { transform: translateX(5px); }
        }
    </style>
</body>
</html>