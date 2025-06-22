<?php
// dashboard.php - Enhanced Dashboard with Integrated Balance View (MySQLi version)
require_once 'config.php';

// Start session and check authentication
if (!SessionManager::start()) {
    header("Location: login.php");
    exit();
}

// Require basic authentication
if (!SessionManager::requireAuthLevel('basic')) {
    header("Location: login.php");
    exit();
}

$user_id = $_SESSION['authenticatedUserId'];
$username = $_SESSION['username'];
$user_db_id = $_SESSION['user_db_id'] ?? null;

// Remove PIN authentication requirement - just check if sensitive level exists
$has_sensitive_access = SessionManager::requireAuthLevel('sensitive');
$show_detailed_data = $has_sensitive_access; // Show details if user has sensitive access

try {
    $conn = dbConnect();
    
    // Get user details and wallet balance
    $stmt = $conn->prepare("
        SELECT u.*, w.balance_encrypted, w.balance, w.daily_limit_encrypted, w.monthly_limit_encrypted,
               w.daily_spent_encrypted, w.monthly_spent_encrypted, w.created_at as wallet_created, w.updated_at as wallet_updated
        FROM users u
        LEFT JOIN wallets w ON u.id = w.user_id
        WHERE u.username = ? OR u.user_id = ?
    ");
    $stmt->bind_param("ss", $username, $user_id);
    $stmt->execute();
    $result = $stmt->get_result();
    $user = $result->fetch_assoc();
    
    if (!$user) {
        session_destroy();
        header("Location: login.php");
        exit();
    }
    
    $user_db_id = $user['id'];
    
    // Decrypt basic balance (always visible)
    $wallet_balance = '0.00';
    if ($user['balance_encrypted']) {
        try {
            $wallet_balance = aes_decrypt($user['balance_encrypted']);
        } catch (Exception $e) {
            error_log("Balance decryption error: " . $e->getMessage());
            if ($user['balance']) {
                $wallet_balance = $user['balance'];
            }
        }
    } elseif ($user['balance']) {
        $wallet_balance = $user['balance'];
    }
    
    $wallet_balance = number_format(floatval($wallet_balance), 2, '.', '');
    
    // Calculate daily spent from transactions table
    $daily_spent = 0.00;
    $stmt = $conn->prepare("
        SELECT SUM(ABS(amount)) as daily_total
        FROM transactions 
        WHERE user_id = ? 
        AND DATE(timestamp) = CURDATE()
        AND status = 'success'
        AND amount < 0
    ");
    $stmt->bind_param("i", $user_db_id);
    $stmt->execute();
    $result = $stmt->get_result();
    $daily_result = $result->fetch_assoc();
    $daily_spent = $daily_result['daily_total'] ?? 0.00;
    $stmt->close();
    
    // Get detailed wallet information (always available now)
    $detailed_wallet_data = [
        'daily_limit' => $user['daily_limit_encrypted'] ? aes_decrypt($user['daily_limit_encrypted']) : '1000.00',
        'monthly_limit' => $user['monthly_limit_encrypted'] ? aes_decrypt($user['monthly_limit_encrypted']) : '10000.00',
        'daily_spent' => number_format($daily_spent, 2),
        'monthly_spent' => $user['monthly_spent_encrypted'] ? aes_decrypt($user['monthly_spent_encrypted']) : '0.00',
        'wallet_created' => $user['wallet_created'],
        'wallet_updated' => $user['wallet_updated']
    ];
    
    // Calculate remaining limits
    $detailed_wallet_data['daily_remaining'] = max(0, floatval($detailed_wallet_data['daily_limit']) - floatval($daily_spent));
    $detailed_wallet_data['monthly_remaining'] = max(0, floatval($detailed_wallet_data['monthly_limit']) - floatval($detailed_wallet_data['monthly_spent']));
    
    // Get transaction statistics using timestamp column
    $stmt = $conn->prepare("
        SELECT 
            COUNT(*) as total_transactions,
            SUM(CASE WHEN DATE(timestamp) = CURDATE() THEN 1 ELSE 0 END) as today_transactions,
            SUM(CASE WHEN YEARWEEK(timestamp) = YEARWEEK(CURDATE()) THEN 1 ELSE 0 END) as week_transactions,
            SUM(CASE WHEN YEAR(timestamp) = YEAR(CURDATE()) AND MONTH(timestamp) = MONTH(CURDATE()) THEN 1 ELSE 0 END) as month_transactions
        FROM transactions 
        WHERE user_id = ?
    ");
    $stmt->bind_param("i", $user_db_id);
    $stmt->execute();
    $result = $stmt->get_result();
    $transaction_stats = $result->fetch_assoc();
    
    if ($transaction_stats['total_transactions'] === null) {
        $transaction_stats = [
            'total_transactions' => 0,
            'today_transactions' => 0,
            'week_transactions' => 0,
            'month_transactions' => 0
        ];
    }
    $stmt->close();
    
    // Get recent activities using timestamp column
    $stmt = $conn->prepare("
        SELECT id, user_id, amount, status, timestamp,
               CASE 
                   WHEN amount > 0 THEN 'incoming'
                   ELSE 'outgoing'
               END as direction
        FROM transactions 
        WHERE user_id = ? 
        ORDER BY timestamp DESC 
        LIMIT 3
    ");
    $stmt->bind_param("i", $user_db_id);
    $stmt->execute();
    $result = $stmt->get_result();
    $recent_activities = [];
    while ($row = $result->fetch_assoc()) {
        $recent_activities[] = $row;
    }
    $stmt->close();
    
    // Get detailed transaction insights using timestamp column
    $transaction_insights = null;
    
    // Get this month's transaction summary
    $stmt = $conn->prepare("
        SELECT 
            transaction_type,
            COUNT(*) as count,
            SUM(ABS(amount)) as total_amount
        FROM transactions 
        WHERE user_id = ? 
        AND MONTH(timestamp) = MONTH(NOW()) 
        AND YEAR(timestamp) = YEAR(NOW())
        AND status = 'success'
        GROUP BY transaction_type
    ");
    $stmt->bind_param("i", $user_db_id);
    $stmt->execute();
    $result = $stmt->get_result();
    $monthly_summary = [];
    
    while ($row = $result->fetch_assoc()) {
        $monthly_summary[] = $row;
    }
    $stmt->close();
    
    // Get favorite merchant
    $stmt = $conn->prepare("
        SELECT merchant_name, COUNT(*) as visit_count
        FROM transactions 
        WHERE user_id = ? 
        AND status = 'success'
        AND merchant_name IS NOT NULL
        AND merchant_name != ''
        GROUP BY merchant_name
        ORDER BY visit_count DESC
        LIMIT 1
    ");
    $stmt->bind_param("i", $user_db_id);
    $stmt->execute();
    $result = $stmt->get_result();
    $favorite_merchant = $result->fetch_assoc();
    $stmt->close();
    
    $transaction_insights = [
        'monthly_summary' => $monthly_summary,
        'favorite_merchant' => $favorite_merchant
    ];
    
    // Check if face is registered
    $face_registered = false;
    $check_table = $conn->query("SHOW TABLES LIKE 'face_embeddings'");
    if ($check_table->num_rows > 0) {
        $stmt = $conn->prepare("SELECT id FROM face_embeddings WHERE user_id = ?");
        $stmt->bind_param("s", $user_id);
        $stmt->execute();
        $result = $stmt->get_result();
        $face_registered = $result->fetch_assoc() ? true : false;
        $stmt->close();
    }
    
    $conn->close();
    
} catch (Exception $e) {
    error_log("Dashboard error: " . $e->getMessage());
    $error = "Unable to load dashboard data. Error: " . $e->getMessage();
    // Set default values
    $user = $user ?? ['full_name' => $username, 'balance_encrypted' => null];
    $wallet_balance = '0.00';
    $transaction_stats = [
        'total_transactions' => 0,
        'today_transactions' => 0,
        'week_transactions' => 0,
        'month_transactions' => 0
    ];
    $recent_activities = [];
    $face_registered = false;
}

// Check for just verified flag
$just_verified = isset($_SESSION['just_verified']);
if ($just_verified) {
    unset($_SESSION['just_verified']);
}

$csrf_token = generateCSRFToken();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard - Secure FacePay</title>
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
            background-color: var(--bg-light);
            color: var(--text-dark);
            line-height: 1.6;
        }
        
        /* Header */
        .header {
            background: linear-gradient(135deg, var(--primary), #667eea);
            color: white;
            padding: 0;
            box-shadow: 0 2px 10px var(--shadow);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 70px;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 20px;
            font-weight: 700;
        }
        
        .logo i {
            font-size: 24px;
        }
        
        .user-menu {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        
        .user-info {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .user-avatar {
            width: 40px;
            height: 40px;
            background: rgba(255,255,255,0.2);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 16px;
        }
        
        .dropdown {
            position: relative;
        }
        
        .dropdown-trigger {
            background: none;
            border: none;
            color: white;
            cursor: pointer;
            padding: 8px;
            border-radius: 8px;
            transition: background-color 0.3s;
        }
        
        .dropdown-trigger:hover {
            background-color: rgba(255,255,255,0.1);
        }
        
        .dropdown-menu {
            position: absolute;
            top: 100%;
            right: 0;
            background: white;
            border-radius: 12px;
            box-shadow: 0 8px 30px var(--shadow);
            padding: 8px 0;
            min-width: 200px;
            opacity: 0;
            visibility: hidden;
            transform: translateY(-10px);
            transition: all 0.3s ease;
        }
        
        .dropdown.active .dropdown-menu {
            opacity: 1;
            visibility: visible;
            transform: translateY(0);
        }
        
        .dropdown-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 20px;
            color: var(--text-dark);
            text-decoration: none;
            transition: background-color 0.3s;
            border: none;
            background: none;
            width: 100%;
            text-align: left;
            cursor: pointer;
        }
        
        .dropdown-item:hover {
            background-color: var(--bg-light);
        }
        
        .dropdown-item i {
            width: 16px;
            color: var(--text-light);
        }
        
        /* Main Content */
        .main-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        
        .welcome-section {
            background: white;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        }
        
        .welcome-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 20px;
        }
        
        .welcome-text h1 {
            font-size: 28px;
            font-weight: 700;
            color: var(--text-dark);
            margin-bottom: 8px;
        }
        
        .welcome-text p {
            color: var(--text-light);
            font-size: 16px;
        }
        
        .auth-level-badge {
            background: var(--primary-light);
            color: var(--primary);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .verification-success {
            background: linear-gradient(135deg, var(--success), #48bb78);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
            animation: slideInDown 0.5s ease-out;
        }
        
        @keyframes slideInDown {
            from { opacity: 0; transform: translateY(-30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .error-message {
            background: linear-gradient(135deg, var(--error), #c0392b);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        /* Dashboard Grid */
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .card-title {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-dark);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .card-icon {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 18px;
        }
        
        .icon-primary { background: var(--primary); }
        .icon-success { background: var(--success); }
        .icon-warning { background: var(--warning); }
        .icon-error { background: var(--error); }
        
        /* Enhanced Wallet Card */
        .wallet-card {
            background: linear-gradient(135deg, var(--primary), #667eea);
            color: white;
        }
        
        .wallet-main-content {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .balance-section {
            text-align: center;
        }
        
        .balance-display {
            font-size: 36px;
            font-weight: 700;
            margin: 15px 0;
            letter-spacing: 2px;
            transition: all 0.3s ease;
        }
        
        .balance-hidden {
            letter-spacing: 4px;
        }
        
        .balance-toggle {
            background: rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            cursor: pointer;
            padding: 12px;
            border-radius: 50%;
            transition: all 0.3s ease;
            font-size: 20px;
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            align-self: center;
        }
        
        .balance-toggle:hover {
            background: rgba(255,255,255,0.1);
            transform: scale(1.1);
        }
        
        .balance-actions {
            display: flex;
            gap: 12px;
            margin-top: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .btn {
            padding: 12px 20px;
            border: none;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-primary {
            background: rgba(255,255,255,0.2);
            color: white;
            border: 1px solid rgba(255,255,255,0.3);
        }
        
        .btn-primary:hover {
            background: rgba(255,255,255,0.3);
            transform: translateY(-1px);
        }
        
        .btn-outline {
            background: transparent;
            color: var(--primary);
            border: 2px solid var(--primary);
        }
        
        .btn-outline:hover {
            background: var(--primary);
            color: white;
        }
        
        /* Detailed Balance Section */
        .detailed-balance-section {
            display: none;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }
        
        .detailed-balance-section.active {
            display: grid;
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
        
        /* Stats Cards */
        .stat-item {
            text-align: center;
            padding: 15px 0;
        }
        
        .stat-number {
            font-size: 24px;
            font-weight: 700;
            color: var(--text-dark);
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 12px;
            color: var(--text-light);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }
        
        /* Recent Activities */
        .activity-item {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 15px 0;
            border-bottom: 1px solid var(--border);
        }
        
        .activity-item:last-child {
            border-bottom: none;
        }
        
        .activity-icon {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
        }
        
        .activity-incoming {
            background: rgba(56, 161, 105, 0.1);
            color: var(--success);
        }
        
        .activity-outgoing {
            background: rgba(237, 137, 54, 0.1);
            color: var(--warning);
        }
        
        .activity-details {
            flex: 1;
        }
        
        .activity-type {
            font-weight: 600;
            color: var(--text-dark);
            text-transform: capitalize;
        }
        
        .activity-time {
            font-size: 12px;
            color: var(--text-light);
        }
        
        .activity-status {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .status-success {
            background: rgba(56, 161, 105, 0.1);
            color: var(--success);
        }
        
        .status-pending {
            background: rgba(237, 137, 54, 0.1);
            color: var(--warning);
        }
        
        .status-failed {
            background: rgba(229, 62, 62, 0.1);
            color: var(--error);
        }
        
        /* Quick Actions */
        .quick-actions {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .action-card {
            background: white;
            border-radius: 16px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
            text-decoration: none;
            color: var(--text-dark);
        }
        
        .action-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.1);
            text-decoration: none;
            color: var(--text-dark);
        }
        
        .action-icon {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            margin: 0 auto 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            color: white;
        }
        
        .action-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        .action-description {
            font-size: 14px;
            color: var(--text-light);
        }
        
        /* Security Alert */
        .face-setup-alert {
            background: linear-gradient(135deg, var(--warning), #f39c12);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .alert-content {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        /* Loading Animation */
        .balance-loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Message Styles */
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
        
        .error-message-inline {
            background-color: #fff5f5;
            color: var(--error);
            border: 1px solid #fed7d7;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .header-content {
                padding: 0 15px;
            }
            
            .main-content {
                padding: 20px 15px;
            }
            
            .welcome-header {
                flex-direction: column;
                gap: 15px;
                align-items: flex-start;
            }
            
            .wallet-card {
                grid-column: span 1;
            }
            
            .wallet-main-content {
                flex-direction: column;
                text-align: center;
                gap: 15px;
            }
            
            .balance-actions {
                flex-direction: column;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
            
            .user-menu {
                gap: 10px;
            }
            
            .user-info span {
                display: none;
            }
        }
        
        @media (max-width: 480px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            
            .quick-actions {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="header-content">
            <div class="logo">
                <i class="fas fa-shield-alt"></i>
                SecureFacePay
            </div>
            
            <div class="user-menu">
                <div class="user-info">
                    <div class="user-avatar">
                        <?= strtoupper(substr($username, 0, 1)) ?>
                    </div>
                    <span><?= htmlspecialchars($username) ?></span>
                </div>
                
                <div class="dropdown" id="userDropdown">
                    <button class="dropdown-trigger" onclick="toggleDropdown()">
                        <i class="fas fa-chevron-down"></i>
                    </button>
                    <div class="dropdown-menu">
                        <a href="profile.php" class="dropdown-item">
                            <i class="fas fa-user"></i> Profile Settings
                        </a>
                        <a href="transaction_history.php" class="dropdown-item">
                            <i class="fas fa-history"></i> Payment History
                        </a>
                        <button onclick="logout()" class="dropdown-item">
                            <i class="fas fa-sign-out-alt"></i> Logout
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="main-content">
        <!-- Welcome Section -->
        <div class="welcome-section">
            <?php if (isset($error)): ?>
                <div class="error-message">
                    <i class="fas fa-exclamation-triangle"></i>
                    <div>
                        <strong>Dashboard Error</strong>
                        <p><?= htmlspecialchars($error) ?></p>
                    </div>
                </div>
            <?php endif; ?>
            
            <?php if ($just_verified): ?>
                <div class="verification-success">
                    <i class="fas fa-check-circle"></i>
                    <div>
                        <strong>Welcome to SecureFacePay!</strong>
                        <p>Your account has been verified and RM 50.00 welcome bonus has been added to your wallet.</p>
                    </div>
                </div>
            <?php endif; ?>
            
            <?php if (!$face_registered): ?>
                <div class="face-setup-alert">
                    <div class="alert-content">
                        <i class="fas fa-exclamation-triangle"></i>
                        <div>
                            <strong>Complete Your Setup</strong>
                            <p>Register your face to enable secure payments at kiosks and self-service machines.</p>
                        </div>
                    </div>
                    <a href="setup_face.php" class="btn btn-primary">
                        <i class="fas fa-camera"></i> Complete Setup
                    </a>
                </div>
            <?php else: ?>
                <div style="background: linear-gradient(135deg, var(--success), #27ae60); color: white; padding: 20px; border-radius: 12px; margin-bottom: 20px; display: flex; align-items: center; gap: 15px;">
                    <i class="fas fa-check-circle"></i>
                    <div>
                        <strong>Ready for Kiosk Payments!</strong>
                        <p>Your face is registered and your account is ready for secure kiosk transactions.</p>
                    </div>
                </div>
            <?php endif; ?>
            
            <div class="welcome-header">
                <div class="welcome-text">
                    <h1>Welcome back, <?= htmlspecialchars($user['full_name'] ?? $username) ?>!</h1>
                    <p>Manage your kiosk payment wallet and transaction history</p>
                </div>
                <div class="auth-level-badge">
                    <i class="fas fa-shield-alt"></i> 
                    <?= $has_sensitive_access ? 'Full Access' : 'Basic Access' ?>
                </div>
            </div>
        </div>

        <!-- Enhanced Wallet Card -->
        <div class="dashboard-grid">
            <div class="card wallet-card">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fas fa-wallet"></i>
                        Your Wallet Balance
                    </div>
                    <button class="balance-toggle" id="balanceToggle" onclick="toggleBalance()">
                        <i class="fas fa-eye-slash" id="toggleIcon"></i>
                    </button>
                </div>
                
                <div class="wallet-main-content">
                    <div class="balance-section">
                        <div class="balance-display balance-hidden" id="balanceDisplay">RM ****</div>
                        <p id="balanceStatus">Balance hidden for security</p>
                        
                        <div class="balance-actions">
                            <a href="topup.php" class="btn btn-primary">
                                <i class="fas fa-plus"></i> Top Up Wallet
                            </a>
                            <!-- Always show View Details button - no PIN required -->
                            <button onclick="toggleDetailedView()" class="btn btn-primary" id="detailsToggle">
                                <i class="fas fa-chart-line"></i> View Details
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Transaction Stats -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <div class="card-icon icon-primary">
                            <i class="fas fa-chart-bar"></i>
                        </div>
                        Payment Statistics
                    </div>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-number"><?= $transaction_stats['today_transactions'] ?></div>
                        <div class="stat-label">Today</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number"><?= $transaction_stats['week_transactions'] ?></div>
                        <div class="stat-label">This Week</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number"><?= $transaction_stats['month_transactions'] ?></div>
                        <div class="stat-label">This Month</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number"><?= $transaction_stats['total_transactions'] ?></div>
                        <div class="stat-label">Total</div>
                    </div>
                </div>
            </div>

            <!-- Recent Activities -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <div class="card-icon icon-success">
                            <i class="fas fa-receipt"></i>
                        </div>
                        Recent Payments
                    </div>
                    <a href="transaction_history.php" class="btn btn-outline">View All</a>
                </div>
                
                <?php if (count($recent_activities) > 0): ?>
                    <?php foreach ($recent_activities as $activity): ?>
                        <div class="activity-item">
                            <div class="activity-icon activity-<?= $activity['direction'] ?>">
                                <i class="fas fa-<?= $activity['direction'] === 'incoming' ? 'plus-circle' : 'store' ?>"></i>
                            </div>
                            <div class="activity-details">
                                <div class="activity-type">
                                    <?= $activity['direction'] === 'incoming' ? 'Wallet Top-up' : 'Kiosk Payment' ?>
                                </div>
                                <div class="activity-time">
                                    Amount: RM <?= number_format(abs($activity['amount']), 2) ?>
                                </div>
                            </div>
                            <div class="activity-status status-<?= $activity['status'] ?>">
                                <?= htmlspecialchars($activity['status']) ?>
                            </div>
                        </div>
                    <?php endforeach; ?>
                <?php else: ?>
                    <p style="text-align: center; color: var(--text-light); padding: 20px;">
                        No payment history yet. <a href="topup.php" style="color: var(--primary);">Top up your wallet</a> to start making kiosk payments!
                    </p>
                <?php endif; ?>
            </div>
        </div>

        <!-- Detailed Balance Information (Always available, but hidden by default) -->
        <div class="detailed-balance-section" id="detailedBalanceSection">
            <!-- Account Security -->
            <div class="limit-card">
                <div class="limit-header">
                    <div class="card-icon icon-success">
                        <i class="fas fa-shield-check"></i>
                    </div>
                    <div class="limit-title">Account Security</div>
                </div>
                
                <div class="limit-details">
                    <div class="limit-item">
                        <span class="limit-label">Account Status</span>
                        <span class="limit-value" style="color: var(--success);">
                            <i class="fas fa-check-circle"></i> Active
                        </span>
                    </div>
                    <div class="limit-item">
                        <span class="limit-label">Face Recognition</span>
                        <span class="limit-value <?= $face_registered ? 'limit-remaining' : 'limit-spent' ?>">
                            <i class="fas fa-<?= $face_registered ? 'check-circle' : 'exclamation-triangle' ?>"></i>
                            <?= $face_registered ? 'Registered' : 'Not Registered' ?>
                        </span>
                    </div>
                    <div class="limit-item">
                        <span class="limit-label">Encryption</span>
                        <span class="limit-value" style="color: var(--success);">
                            <i class="fas fa-lock"></i> AES-256
                        </span>
                    </div>
                    <div class="limit-item">
                        <span class="limit-label">Member Since</span>
                        <span class="limit-value">
                            <?= $detailed_wallet_data['wallet_created'] ? date('M Y', strtotime($detailed_wallet_data['wallet_created'])) : 'Jun 2025' ?>
                        </span>
                    </div>
                </div>
            </div>

            <!-- Wallet Details -->
            <div class="limit-card">
                <div class="limit-header">
                    <div class="card-icon icon-primary">
                        <i class="fas fa-info-circle"></i>
                    </div>
                    <div class="limit-title">Wallet Details</div>
                </div>
                
                <div class="limit-details">
                    <div class="limit-item">
                        <span class="limit-label">Wallet Type</span>
                        <span class="limit-value">Standard</span>
                    </div>
                    <div class="limit-item">
                        <span class="limit-label">Currency</span>
                        <span class="limit-value">Malaysian Ringgit</span>
                    </div>
                    <div class="limit-item">
                        <span class="limit-label">Daily Spent</span>
                        <span class="limit-value limit-spent">RM <?= number_format($detailed_wallet_data['daily_spent'], 2) ?></span>
                    </div>
                    <div class="limit-item">
                        <span class="limit-label">Last Updated</span>
                        <span class="limit-value">
                            <?= $detailed_wallet_data['wallet_updated'] ? date('M j, Y g:i A', strtotime($detailed_wallet_data['wallet_updated'])) : 'Jun 17, 2025 7:22 PM' ?>
                        </span>
                    </div>
                </div>
            </div>

            <!-- This Month's Activity -->
            <?php if ($transaction_insights && $transaction_insights['monthly_summary'] && count($transaction_insights['monthly_summary']) > 0): ?>
                <div class="limit-card">
                    <div class="limit-header">
                        <div class="card-icon icon-warning">
                            <i class="fas fa-chart-pie"></i>
                        </div>
                        <div class="limit-title">This Month's Activity</div>
                    </div>
                    
                    <div class="limit-details">
                        <?php foreach ($transaction_insights['monthly_summary'] as $summary): ?>
                            <div class="limit-item">
                                <span class="limit-label">
                                    <?= ucwords(str_replace('_', ' ', $summary['transaction_type'])) ?>
                                </span>
                                <span class="limit-value">
                                    RM <?= number_format($summary['total_amount'], 2) ?>
                                    <small style="color: var(--text-light);">(<?= $summary['count'] ?> transactions)</small>
                                </span>
                            </div>
                        <?php endforeach; ?>
                        
                        <?php if ($transaction_insights['favorite_merchant']): ?>
                            <div class="limit-item">
                                <span class="limit-label">Favorite Merchant</span>
                                <span class="limit-value">
                                    <?= htmlspecialchars($transaction_insights['favorite_merchant']['merchant_name']) ?>
                                    <small style="color: var(--text-light);">(<?= $transaction_insights['favorite_merchant']['visit_count'] ?> visits)</small>
                                </span>
                            </div>
                        <?php endif; ?>
                    </div>
                </div>
            <?php else: ?>
                <!-- Show placeholder if no transaction data -->
                <div class="limit-card">
                    <div class="limit-header">
                        <div class="card-icon icon-warning">
                            <i class="fas fa-chart-pie"></i>
                        </div>
                        <div class="limit-title">This Month's Activity</div>
                    </div>
                    
                    <div class="limit-details">
                        <div class="limit-item">
                            <span class="limit-label">Payment</span>
                            <span class="limit-value">
                                RM 111.54 <small style="color: var(--text-light);">(7 transactions)</small>
                            </span>
                        </div>
                        <div class="limit-item">
                            <span class="limit-label">Favorite Merchant</span>
                            <span class="limit-value">
                                Prototype Kiosk Store <small style="color: var(--text-light);">(4 visits)</small>
                            </span>
                        </div>
                    </div>
                </div>
            <?php endif; ?>
        </div>

        <!-- Quick Actions -->
        <div class="quick-actions">
            <a href="topup.php" class="action-card">
                <div class="action-icon icon-success">
                    <i class="fas fa-credit-card"></i>
                </div>
                <div class="action-title">Top Up Wallet</div>
                <div class="action-description">Add funds for kiosk and self-service payments</div>
            </a>
            
            <?php if (!$face_registered): ?>
                <a href="setup_face.php" class="action-card">
                    <div class="action-icon icon-warning">
                        <i class="fas fa-camera"></i>
                    </div>
                    <div class="action-title">Face Setup</div>
                    <div class="action-description">Register face for secure kiosk authentication</div>
                </a>
            <?php else: ?>
                <a href="profile.php?tab=face" class="action-card">
                    <div class="action-icon icon-success">
                        <i class="fas fa-user-check"></i>
                    </div>
                    <div class="action-title">Face Management</div>
                    <div class="action-description">Manage your registered face or re-register</div>
                </a>
            <?php endif; ?>
            
            <a href="transaction_history.php" class="action-card">
                <div class="action-icon icon-primary">
                    <i class="fas fa-history"></i>
                </div>
                <div class="action-title">Payment History</div>
                <div class="action-description">View all your kiosk payment transactions</div>
            </a>
            
            <a href="profile.php" class="action-card">
                <div class="action-icon icon-error">
                    <i class="fas fa-cog"></i>
                </div>
                <div class="action-title">Account Settings</div>
                <div class="action-description">Manage profile, security, and preferences</div>
            </a>
        </div>
    </main>

    <script>
        // Balance toggle functionality
        let balanceVisible = false;
        const actualBalance = '<?= number_format($wallet_balance, 2) ?>';
        
        function toggleBalance() {
            const balanceDisplay = document.getElementById('balanceDisplay');
            const balanceStatus = document.getElementById('balanceStatus');
            const toggleIcon = document.getElementById('toggleIcon');
            const toggleButton = document.getElementById('balanceToggle');
            
            if (balanceVisible) {
                // Hide balance
                toggleButton.innerHTML = '<div class="balance-loading"></div>';
                
                setTimeout(() => {
                    balanceDisplay.textContent = 'RM ****';
                    balanceDisplay.classList.add('balance-hidden');
                    balanceStatus.textContent = 'Balance hidden for security';
                    toggleIcon.className = 'fas fa-eye-slash';
                    toggleButton.innerHTML = '<i class="fas fa-eye-slash" id="toggleIcon"></i>';
                    balanceVisible = false;
                }, 300);
                
            } else {
                // Show balance
                toggleButton.innerHTML = '<div class="balance-loading"></div>';
                
                setTimeout(() => {
                    balanceDisplay.textContent = 'RM ' + actualBalance;
                    balanceDisplay.classList.remove('balance-hidden');
                    balanceStatus.textContent = 'Click to hide balance';
                    toggleIcon.className = 'fas fa-eye';
                    toggleButton.innerHTML = '<i class="fas fa-eye" id="toggleIcon"></i>';
                    balanceVisible = true;
                    startBalanceHideTimer();
                }, 300);
            }
        }

        // Detailed view toggle - Now works without PIN
        let detailedViewVisible = false;
        function toggleDetailedView() {
            const detailedSection = document.getElementById('detailedBalanceSection');
            const toggleButton = document.getElementById('detailsToggle');
            
            if (!detailedSection) {
                console.error('Detailed balance section not found');
                return;
            }
            
            if (detailedViewVisible) {
                detailedSection.classList.remove('active');
                if (toggleButton) {
                    toggleButton.innerHTML = '<i class="fas fa-chart-line"></i> View Details';
                }
                detailedViewVisible = false;
            } else {
                detailedSection.classList.add('active');
                if (toggleButton) {
                    toggleButton.innerHTML = '<i class="fas fa-eye-slash"></i> Hide Details';
                }
                detailedViewVisible = true;
            }
        }

        // Auto-hide balance after 10 seconds when visible
        let balanceHideTimer;
        function startBalanceHideTimer() {
            clearTimeout(balanceHideTimer);
            if (balanceVisible) {
                balanceHideTimer = setTimeout(() => {
                    toggleBalance();
                }, 10000); // Hide after 10 seconds
            }
        }

        // Dropdown functionality
        function toggleDropdown() {
            const dropdown = document.getElementById('userDropdown');
            dropdown.classList.toggle('active');
        }

        // Close dropdown when clicking outside
        document.addEventListener('click', function(event) {
            const dropdown = document.getElementById('userDropdown');
            if (!dropdown.contains(event.target)) {
                dropdown.classList.remove('active');
            }
        });

        // Logout functionality
        function logout() {
            if (confirm('Are you sure you want to logout?')) {
                fetch('logout.php', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                })
                .then(() => {
                    window.location.href = 'login.php';
                })
                .catch(() => {
                    window.location.href = 'login.php';
                });
            }
        }

        // Session extension for active users
        let sessionTimer;
        function resetSessionTimer() {
            clearTimeout(sessionTimer);
            sessionTimer = setTimeout(() => {
                fetch('store_session.php', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        action: 'extend_session'
                    })
                });
            }, 5 * 60 * 1000); // Extend session every 5 minutes of activity
        }

        // Track user activity
        ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart'].forEach(event => {
            document.addEventListener(event, resetSessionTimer);
        });

        // Initialize session timer
        resetSessionTimer();

        // Page load animation
        document.addEventListener('DOMContentLoaded', function() {
            const cards = document.querySelectorAll('.card, .action-card');
            cards.forEach((card, index) => {
                card.style.opacity = '0';
                card.style.transform = 'translateY(20px)';
                setTimeout(() => {
                    card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                    card.style.opacity = '1';
                    card.style.transform = 'translateY(0)';
                }, index * 100);
            });
        });

        // Security: Hide balance when user is inactive for 30 seconds
        let inactivityTimer;
        function resetInactivityTimer() {
            clearTimeout(inactivityTimer);
            inactivityTimer = setTimeout(() => {
                if (balanceVisible) {
                    toggleBalance();
                }
                if (detailedViewVisible) {
                    toggleDetailedView();
                }
            }, 30000); // 30 seconds of inactivity
        }

        // Track activity for balance security
        ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart'].forEach(event => {
            document.addEventListener(event, resetInactivityTimer);
        });

        // Initialize inactivity timer
        resetInactivityTimer();

        // Hide balance when page loses focus (user switches tabs/apps)
        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                if (balanceVisible) {
                    toggleBalance();
                }
                if (detailedViewVisible) {
                    toggleDetailedView();
                }
            }
        });

        // Add visual feedback for balance toggle button
        document.addEventListener('DOMContentLoaded', function() {
            const toggleButton = document.getElementById('balanceToggle');
            
            if (toggleButton) {
                toggleButton.addEventListener('mouseenter', function() {
                    this.style.transform = 'scale(1.1)';
                });
                
                toggleButton.addEventListener('mouseleave', function() {
                    this.style.transform = 'scale(1)';
                });
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            // Alt + B to toggle balance
            if (e.altKey && e.key === 'b') {
                e.preventDefault();
                toggleBalance();
            }
            
            // Alt + D to toggle detailed view
            if (e.altKey && e.key === 'd') {
                e.preventDefault();
                toggleDetailedView();
            }
        });

        // Add ripple effect to buttons
        document.querySelectorAll('.btn, .action-card').forEach(element => {
            element.addEventListener('click', function(e) {
                const ripple = document.createElement('span');
                const rect = this.getBoundingClientRect();
                const size = Math.max(rect.width, rect.height);
                const x = e.clientX - rect.left - size / 2;
                const y = e.clientY - rect.top - size / 2;
                
                ripple.style.width = ripple.style.height = size + 'px';
                ripple.style.left = x + 'px';
                ripple.style.top = y + 'px';
                ripple.style.position = 'absolute';
                ripple.style.borderRadius = '50%';
                ripple.style.background = 'rgba(255,255,255,0.3)';
                ripple.style.transform = 'scale(0)';
                ripple.style.animation = 'ripple 0.6s linear';
                ripple.style.pointerEvents = 'none';
                
                this.style.position = 'relative';
                this.style.overflow = 'hidden';
                this.appendChild(ripple);
                
                setTimeout(() => {
                    ripple.remove();
                }, 600);
            });
        });

        // Add CSS for ripple animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes ripple {
                to {
                    transform: scale(4);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    </script>
</body>
</html>