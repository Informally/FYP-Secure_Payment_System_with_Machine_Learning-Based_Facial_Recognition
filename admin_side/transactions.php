<?php
require_once 'config/admin_config.php';

// Require authentication
$admin = requireAdminAuth(['super_admin', 'admin', 'finance']);

$message = '';
$message_type = '';

// Handle form submissions for transaction actions
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $action = $_POST['action'] ?? '';
    
    try {
        $conn = dbConnect();
        
        if ($action === 'process_refund') {
            $transaction_id = intval($_POST['transaction_id']);
            $refund_reason = trim($_POST['refund_reason']) ?: 'Admin processed refund';
            
            // Get original transaction details
            $stmt = $conn->prepare("SELECT * FROM transactions WHERE id = ? AND status = 'success'");
            $stmt->bind_param("i", $transaction_id);
            $stmt->execute();
            $transaction = $stmt->get_result()->fetch_assoc();
            $stmt->close();
            
            if ($transaction) {
                // Start database transaction for atomic operations
                $conn->begin_transaction();
                
                try {
                    // Get current wallet balance
                    $wallet_stmt = $conn->prepare("
                        SELECT balance_encrypted, balance 
                        FROM wallets 
                        WHERE user_id = ? FOR UPDATE
                    ");
                    $wallet_stmt->bind_param("i", $transaction['user_id']);
                    $wallet_stmt->execute();
                    $wallet_result = $wallet_stmt->get_result();
                    $wallet_data = $wallet_result->fetch_assoc();
                    $wallet_stmt->close();
                    
                    // Calculate current balance (prioritize encrypted version)
                    $current_balance = '0.00';
                    if ($wallet_data && $wallet_data['balance_encrypted']) {
                        try {
                            $current_balance = aes_decrypt($wallet_data['balance_encrypted']);
                        } catch (Exception $e) {
                            $current_balance = $wallet_data['balance'] ?? '0.00';
                        }
                    } elseif ($wallet_data && $wallet_data['balance']) {
                        $current_balance = $wallet_data['balance'];
                    }
                    
                    // Calculate new balance after refund (full amount - no fees deducted)
                    $refund_amount = abs($transaction['amount']);
                    $old_balance = floatval($current_balance);
                    $new_balance = $old_balance + $refund_amount;
                    
                    // Encrypt the new balance
                    $encrypted_balance = aes_encrypt(number_format($new_balance, 2, '.', ''));
                    $balance_plain = number_format($new_balance, 2, '.', '');
                    
                    // Update wallet with both encrypted and plain balance
                    if ($wallet_data) {
                        // Update existing wallet
                        $update_wallet_stmt = $conn->prepare("
                            UPDATE wallets 
                            SET balance_encrypted = ?, balance = ?, updated_at = NOW() 
                            WHERE user_id = ?
                        ");
                        $update_wallet_stmt->bind_param("sdi", $encrypted_balance, $balance_plain, $transaction['user_id']);
                        if (!$update_wallet_stmt->execute()) {
                            throw new Exception("Failed to update wallet: " . $update_wallet_stmt->error);
                        }
                        $update_wallet_stmt->close();
                    } else {
                        // Create new wallet if it doesn't exist
                        $create_wallet_stmt = $conn->prepare("
                            INSERT INTO wallets (user_id, balance_encrypted, balance, created_at, updated_at) 
                            VALUES (?, ?, ?, NOW(), NOW())
                        ");
                        $create_wallet_stmt->bind_param("isd", $transaction['user_id'], $encrypted_balance, $balance_plain);
                        if (!$create_wallet_stmt->execute()) {
                            throw new Exception("Failed to create wallet: " . $create_wallet_stmt->error);
                        }
                        $create_wallet_stmt->close();
                    }
                    
                    // Update original transaction status
                    $update_stmt = $conn->prepare("UPDATE transactions SET status = 'refunded' WHERE id = ?");
                    $update_stmt->bind_param("i", $transaction_id);
                    $update_stmt->execute();
                    $update_stmt->close();
                    
                    // Create refund transaction record
                    $refund_stmt = $conn->prepare("
                        INSERT INTO transactions (
                            user_id, merchant_id, merchant_name, amount, status, 
                            payment_method, transaction_type, timestamp
                        ) VALUES (?, ?, ?, ?, 'success', 'refund', 'refund', NOW())
                    ");
                    $refund_stmt->bind_param("issd", 
                        $transaction['user_id'], 
                        $transaction['merchant_id'], 
                        $transaction['merchant_name'],
                        $refund_amount
                    );
                    $refund_stmt->execute();
                    $refund_stmt->close();
                    
                    // Commit transaction
                    $conn->commit();
                    
                    // Log action using new security system
                    if (function_exists('logAdminSecurityEvent')) {
                        logAdminSecurityEvent($admin['username'], 'process_refund', 'success', [
                            'transaction_id' => $transaction_id,
                            'refund_amount' => $refund_amount,
                            'reason' => $refund_reason,
                            'old_balance' => $old_balance,
                            'new_balance' => $new_balance,
                            'admin_id' => $admin['id'],
                            'user_id' => $transaction['user_id']
                        ]);
                    }
                    
                    $message = "Refund processed successfully! Amount RM " . number_format($refund_amount, 2) . " has been credited to user's wallet. New balance: RM " . number_format($new_balance, 2);
                    $message_type = 'success';
                    
                } catch (Exception $e) {
                    // Rollback transaction on error
                    $conn->rollback();
                    error_log("Refund processing error: " . $e->getMessage());
                    $message = "Error processing refund: " . $e->getMessage();
                    $message_type = 'error';
                }
            } else {
                $message = "Transaction not found or already refunded.";
                $message_type = 'error';
            }
        }
        
        $conn->close();
        
    } catch (Exception $e) {
        $message = "Error: " . $e->getMessage();
        $message_type = 'error';
    }
}

// Get filter parameters
$date_from = $_GET['date_from'] ?? date('Y-m-d', strtotime('-30 days'));
$date_to = $_GET['date_to'] ?? date('Y-m-d');
$merchant_filter = $_GET['merchant'] ?? '';
$status_filter = $_GET['status'] ?? '';
$type_filter = $_GET['type'] ?? '';
$search = $_GET['search'] ?? '';
$page = max(1, intval($_GET['page'] ?? 1));
$per_page = 50;
$offset = ($page - 1) * $per_page;

// Build query
$conn = dbConnect();
$where_conditions = ["DATE(t.timestamp) BETWEEN ? AND ?"];
$params = [$date_from, $date_to];
$param_types = 'ss';

if (!empty($merchant_filter)) {
    $where_conditions[] = "t.merchant_id = ?";
    $params[] = $merchant_filter;
    $param_types .= 's';
}

if (!empty($status_filter)) {
    $where_conditions[] = "t.status = ?";
    $params[] = $status_filter;
    $param_types .= 's';
}

if (!empty($type_filter)) {
    $where_conditions[] = "t.transaction_type = ?";
    $params[] = $type_filter;
    $param_types .= 's';
}

if (!empty($search)) {
    $where_conditions[] = "(u.full_name LIKE ? OR u.email LIKE ? OR t.transaction_id LIKE ? OR t.order_id LIKE ?)";
    $search_param = "%{$search}%";
    $params = array_merge($params, [$search_param, $search_param, $search_param, $search_param]);
    $param_types .= 'ssss';
}

$where_clause = 'WHERE ' . implode(' AND ', $where_conditions);

// Get total count
$count_query = "
    SELECT COUNT(*) as total
    FROM transactions t
    LEFT JOIN users u ON t.user_id = u.id
    {$where_clause}
";

$count_stmt = $conn->prepare($count_query);
$count_stmt->bind_param($param_types, ...$params);
$count_stmt->execute();
$total_transactions = $count_stmt->get_result()->fetch_assoc()['total'];
$count_stmt->close();

// Get transactions with user info
$transactions_query = "
    SELECT 
        t.*,
        u.full_name,
        u.email,
        u.user_id as user_string_id
    FROM transactions t
    LEFT JOIN users u ON t.user_id = u.id
    {$where_clause}
    ORDER BY t.timestamp DESC
    LIMIT ? OFFSET ?
";

// Add pagination params
$params[] = $per_page;
$params[] = $offset;
$param_types .= 'ii';

$transactions_stmt = $conn->prepare($transactions_query);
$transactions_stmt->bind_param($param_types, ...$params);
$transactions_stmt->execute();
$transactions = $transactions_stmt->get_result()->fetch_all(MYSQLI_ASSOC);
$transactions_stmt->close();

// Get summary stats for the date range (SIMPLIFIED - NO PLATFORM FEES)
$stats_query = "
    SELECT 
        COUNT(*) as total_transactions,
        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_transactions,
        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_transactions,
        SUM(CASE WHEN status = 'success' AND transaction_type = 'payment' THEN ABS(amount) ELSE 0 END) as total_revenue,
        COUNT(DISTINCT CASE WHEN merchant_id IS NOT NULL THEN merchant_id END) as active_merchants,
        AVG(CASE WHEN status = 'success' AND transaction_type = 'payment' THEN ABS(amount) ELSE NULL END) as avg_transaction_amount
    FROM transactions
    WHERE DATE(timestamp) BETWEEN ? AND ?
";

$stats_stmt = $conn->prepare($stats_query);
$stats_stmt->bind_param("ss", $date_from, $date_to);
$stats_stmt->execute();
$stats = $stats_stmt->get_result()->fetch_assoc();
$stats_stmt->close();

// Get merchant list for filter dropdown
$merchants_query = "SELECT DISTINCT merchant_id, merchant_name FROM transactions WHERE merchant_id IS NOT NULL ORDER BY merchant_name";
$merchants = $conn->query($merchants_query)->fetch_all(MYSQLI_ASSOC);

// Get daily revenue for chart (last 7 days)
$chart_query = "
    SELECT 
        DATE(timestamp) as date,
        SUM(CASE WHEN status = 'success' AND transaction_type = 'payment' THEN ABS(amount) ELSE 0 END) as revenue,
        COUNT(CASE WHEN status = 'success' THEN 1 ELSE NULL END) as transactions
    FROM transactions
    WHERE DATE(timestamp) >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
    GROUP BY DATE(timestamp)
    ORDER BY date ASC
";
$chart_data = $conn->query($chart_query)->fetch_all(MYSQLI_ASSOC);

$conn->close();

$total_pages = ceil($total_transactions / $per_page);
$success_rate = $stats['total_transactions'] > 0 ? ($stats['successful_transactions'] / $stats['total_transactions']) * 100 : 0;
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transaction Management - FacePay Admin</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        :root {
            --primary: #667eea;
            --primary-dark: #5a67d8;
            --secondary: #764ba2;
            --success: #38a169;
            --warning: #ed8936;
            --error: #e53e3e;
            --info: #3182ce;
            --text-dark: #1a202c;
            --text-medium: #4a5568;
            --text-light: #718096;
            --bg-light: #f7fafc;
            --border: #e2e8f0;
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
        .admin-header {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 0;
            box-shadow: 0 4px 15px var(--shadow);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-content {
            max-width: 1400px;
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
            font-size: 24px;
            font-weight: 700;
        }

        .admin-nav {
            display: flex;
            align-items: center;
            gap: 30px;
        }

        .nav-item {
            color: white;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 8px;
            transition: background-color 0.3s;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .nav-item:hover, .nav-item.active {
            background-color: rgba(255,255,255,0.1);
        }

        .user-menu {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .user-info {
            display: flex;
            align-items: center;
            gap: 10px;
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
        }

        .role-badge {
            background: rgba(255,255,255,0.2);
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Main Content */
        .main-content {
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px 20px;
        }

        .page-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
        }

        .page-title {
            font-size: 32px;
            font-weight: 700;
            color: var(--text-dark);
        }

        .page-subtitle {
            color: var(--text-light);
            margin-top: 5px;
        }

        .export-btn {
            background: linear-gradient(135deg, var(--info), #2980b9);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .export-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(52, 152, 219, 0.3);
        }

        /* Messages */
        .message {
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 500;
        }

        .message.success {
            background: #f0fff4;
            color: var(--success);
            border: 1px solid #c6f6d5;
        }

        .message.error {
            background: #fff5f5;
            color: var(--error);
            border: 1px solid #fed7d7;
        }

        /* Stats Cards */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            text-align: center;
            border-left: 4px solid transparent;
        }

        .stat-card.primary { border-left-color: var(--primary); }
        .stat-card.success { border-left-color: var(--success); }
        .stat-card.warning { border-left-color: var(--warning); }
        .stat-card.error { border-left-color: var(--error); }
        .stat-card.info { border-left-color: var(--info); }

        .stat-value {
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .stat-value.primary { color: var(--primary); }
        .stat-value.success { color: var(--success); }
        .stat-value.warning { color: var(--warning); }
        .stat-value.error { color: var(--error); }
        .stat-value.info { color: var(--info); }

        .stat-label {
            color: var(--text-light);
            font-size: 12px;
        }

        /* Chart Section */
        .chart-section {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        }

        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .chart-title {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-dark);
        }

        /* Filters */
        .filters-section {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        }

        .filters-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            align-items: end;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-label {
            margin-bottom: 5px;
            font-weight: 600;
            color: var(--text-dark);
            font-size: 14px;
        }

        .form-input, .form-select {
            padding: 10px 12px;
            border: 2px solid var(--border);
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s ease;
        }

        .form-input:focus, .form-select:focus {
            outline: none;
            border-color: var(--primary);
        }

        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .btn-primary {
            background: var(--primary);
            color: white;
        }

        .btn-success {
            background: var(--success);
            color: white;
        }

        .btn-warning {
            background: var(--warning);
            color: white;
        }

        .btn-error {
            background: var(--error);
            color: white;
        }

        .btn-info {
            background: var(--info);
            color: white;
        }

        .btn:hover {
            transform: translateY(-1px);
            opacity: 0.9;
        }

        /* Transactions Table */
        .transactions-section {
            background: white;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            overflow: hidden;
        }

        .section-header {
            padding: 20px 25px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .section-title {
            font-size: 20px;
            font-weight: 600;
            color: var(--text-dark);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .transactions-table {
            width: 100%;
            border-collapse: collapse;
        }

        .transactions-table th {
            background: var(--bg-light);
            padding: 15px;
            text-align: left;
            font-weight: 600;
            color: var(--text-medium);
            border-bottom: 1px solid var(--border);
            font-size: 13px;
        }

        .transactions-table td {
            padding: 12px 15px;
            border-bottom: 1px solid var(--border);
            font-size: 14px;
        }

        .transactions-table tbody tr:hover {
            background: #f0f4fa;
        }

        .transaction-id {
            font-family: monospace;
            font-size: 12px;
            color: var(--primary);
            font-weight: 600;
        }

        .user-info-small {
            display: flex;
            flex-direction: column;
        }

        .user-name {
            font-weight: 600;
            color: var(--text-dark);
            margin-bottom: 2px;
        }

        .user-email {
            font-size: 12px;
            color: var(--text-light);
        }

        .merchant-name {
            font-weight: 600;
            color: var(--text-dark);
        }

        .merchant-id {
            font-size: 11px;
            color: var(--text-light);
        }

        .amount-positive {
            color: var(--success);
            font-weight: 600;
        }

        .amount-negative {
            color: var(--error);
            font-weight: 600;
        }

        .status-badge {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }

        .status-badge.success {
            background: rgba(56, 161, 105, 0.1);
            color: var(--success);
        }

        .status-badge.failed {
            background: rgba(229, 62, 62, 0.1);
            color: var(--error);
        }

        .status-badge.pending {
            background: rgba(237, 137, 54, 0.1);
            color: var(--warning);
        }

        .status-badge.refunded {
            background: rgba(113, 128, 150, 0.1);
            color: var(--text-light);
        }

        .transaction-date {
            font-size: 13px;
            color: var(--text-medium);
        }

        .action-buttons {
            display: flex;
            gap: 5px;
        }

        .action-buttons .btn {
            padding: 6px 10px;
            font-size: 11px;
            border-radius: 4px;
        }

        /* Pagination */
        .pagination {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            padding: 20px;
            background: white;
        }

        .pagination a, .pagination span {
            padding: 8px 12px;
            border-radius: 6px;
            text-decoration: none;
            color: var(--text-medium);
            font-weight: 500;
        }

        .pagination a:hover {
            background: var(--bg-light);
        }

        .pagination .current {
            background: var(--primary);
            color: white;
        }

        /* Modal */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
        }

        .modal.show {
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .modal-content {
            background: white;
            border-radius: 16px;
            max-width: 500px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }

        .modal-header {
            padding: 20px 25px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .modal-title {
            font-size: 20px;
            font-weight: 600;
            color: var(--text-dark);
        }

        .modal-close {
            background: none;
            border: none;
            font-size: 24px;
            color: var(--text-light);
            cursor: pointer;
        }

        .modal-body {
            padding: 25px;
        }

        .modal-footer {
            padding: 20px 25px;
            border-top: 1px solid var(--border);
            display: flex;
            justify-content: flex-end;
            gap: 12px;
        }

        .btn-save {
            background: var(--success);
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
        }

        .btn-cancel {
            background: var(--text-light);
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .header-content {
                padding: 0 15px;
            }

            .admin-nav {
                display: none;
            }

            .main-content {
                padding: 20px 15px;
            }

            .filters-grid {
                grid-template-columns: 1fr;
            }

            .transactions-table {
                font-size: 12px;
            }

            .transactions-table th,
            .transactions-table td {
                padding: 8px;
            }

            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <header class="admin-header">
        <div class="header-content">
            <div class="logo">
                <i class="fas fa-shield-alt"></i>
                FacePay Admin
            </div>

            <nav class="admin-nav">
                <a href="dashboard.php" class="nav-item">
                    <i class="fas fa-tachometer-alt"></i> Dashboard
                </a>
                <a href="merchants.php" class="nav-item">
                    <i class="fas fa-store"></i> Merchants
                </a>
                <a href="users.php" class="nav-item">
                    <i class="fas fa-users"></i> Users
                </a>
                <a href="transactions.php" class="nav-item active">
                    <i class="fas fa-credit-card"></i> Transactions
                </a>
            </nav>

            <div class="user-menu">
                <div class="user-info">
                    <div class="user-avatar">
                        <?= strtoupper(substr($admin['full_name'], 0, 1)) ?>
                    </div>
                    <div>
                        <div style="font-size: 14px; font-weight: 600;">
                            <?= htmlspecialchars($admin['full_name']) ?>
                        </div>
                        <div class="role-badge">
                            <?= htmlspecialchars($admin['role']) ?>
                        </div>
                    </div>
                </div>
                <a href="logout.php" class="nav-item">
                    <i class="fas fa-sign-out-alt"></i>
                </a>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="main-content">
        <div class="page-header">
            <div>
                <h1 class="page-title">Transaction Management</h1>
                <p class="page-subtitle">Monitor payments, revenue, and financial operations</p>
            </div>
            <button class="export-btn" onclick="exportMerchantReport()" style="background: linear-gradient(135deg, var(--success), #27ae60);">
                <i class="fas fa-chart-line"></i> Merchant Report
            </button>
        </div>

        <?php if ($message): ?>
            <div class="message <?= $message_type ?>">
                <i class="fas fa-<?= $message_type === 'success' ? 'check-circle' : 'exclamation-triangle' ?>"></i>
                <?= htmlspecialchars($message) ?>
            </div>
        <?php endif; ?>

        <!-- Stats (SIMPLIFIED - NO PLATFORM FEES) -->
        <div class="stats-grid">
            <div class="stat-card success">
                <div class="stat-value success">RM <?= number_format($stats['total_revenue'], 2) ?></div>
                <div class="stat-label">Total Revenue</div>
            </div>
            <div class="stat-card primary">
                <div class="stat-value primary"><?= number_format($stats['total_transactions']) ?></div>
                <div class="stat-label">Total Transactions</div>
            </div>
            <div class="stat-card info">
                <div class="stat-value info"><?= number_format($success_rate, 1) ?>%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-card info">
                <div class="stat-value info"><?= $stats['active_merchants'] ?></div>
                <div class="stat-label">Active Merchants</div>
            </div>
            <div class="stat-card primary">
                <div class="stat-value primary">RM <?= number_format($stats['avg_transaction_amount'] ?? 0, 2) ?></div>
                <div class="stat-label">Avg Transaction</div>
            </div>
        </div>

        <!-- Revenue Chart -->
        <div class="chart-section">
            <div class="chart-header">
                <h3 class="chart-title">
                    <i class="fas fa-chart-line"></i> Daily Revenue Trend (Last 7 Days)
                </h3>
            </div>
            <canvas id="revenueChart" width="400" height="100"></canvas>
        </div>

        <!-- Filters -->
        <div class="filters-section">
            <form method="GET" class="filters-grid">
                <div class="form-group">
                    <label class="form-label">From Date</label>
                    <input type="date" class="form-input" name="date_from" 
                           value="<?= htmlspecialchars($date_from) ?>">
                </div>
                <div class="form-group">
                    <label class="form-label">To Date</label>
                    <input type="date" class="form-input" name="date_to" 
                           value="<?= htmlspecialchars($date_to) ?>">
                </div>
                <div class="form-group">
                    <label class="form-label">Merchant</label>
                    <select class="form-select" name="merchant">
                        <option value="">All Merchants</option>
                        <?php foreach ($merchants as $merchant): ?>
                            <option value="<?= htmlspecialchars($merchant['merchant_id']) ?>" 
                                    <?= $merchant_filter === $merchant['merchant_id'] ? 'selected' : '' ?>>
                                <?= htmlspecialchars($merchant['merchant_name']) ?>
                            </option>
                        <?php endforeach; ?>
                    </select>
                </div>
                <div class="form-group">
                    <label class="form-label">Type</label>
                    <select class="form-select" name="type">
                        <option value="">All Types</option>
                        <option value="payment" <?= $type_filter === 'payment' ? 'selected' : '' ?>>Payment</option>
                        <option value="topup" <?= $type_filter === 'topup' ? 'selected' : '' ?>>Top-up</option>
                        <option value="refund" <?= $type_filter === 'refund' ? 'selected' : '' ?>>Refund</option>
                        <option value="Welcome Bonus" <?= $type_filter === 'Welcome Bonus' ? 'selected' : '' ?>>Welcome Bonus</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="form-label">Search</label>
                    <input type="text" class="form-input" name="search" 
                           value="<?= htmlspecialchars($search) ?>" 
                           placeholder="User, email, transaction ID...">
                </div>
                <div class="form-group">
                    <button type="submit" class="btn btn-primary">
                        <i class="fas fa-filter"></i> Apply Filters
                    </button>
                </div>
            </form>
        </div>

        <!-- Transactions Table -->
        <div class="transactions-section">
            <div class="section-header">
                <div class="section-title">
                    <i class="fas fa-list"></i>
                    Transactions (<?= number_format($total_transactions) ?> found)
                </div>
            </div>

            <table class="transactions-table">
                <thead>
                    <tr>
                        <th>Transaction ID</th>
                        <th>User</th>
                        <th>Merchant</th>
                        <th>Amount</th>
                        <th>Type</th>
                        <th>Status</th>
                        <th>Date</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    <?php if (empty($transactions)): ?>
                        <tr>
                            <td colspan="8" style="text-align: center; padding: 40px; color: var(--text-light);">
                                <i class="fas fa-search" style="font-size: 48px; margin-bottom: 15px; opacity: 0.3;"></i>
                                <br>No transactions found matching your criteria.
                            </td>
                        </tr>
                    <?php else: ?>
                        <?php foreach ($transactions as $transaction): ?>
                            <tr>
                                <td>
                                    <div class="transaction-id">
                                        #<?= $transaction['id'] ?>
                                    </div>
                                    <?php if ($transaction['order_id']): ?>
                                        <div style="font-size: 11px; color: var(--text-light);">
                                            Order: <?= htmlspecialchars($transaction['order_id']) ?>
                                        </div>
                                    <?php endif; ?>
                                </td>
                                <td>
                                    <div class="user-info-small">
                                        <div class="user-name"><?= htmlspecialchars($transaction['full_name'] ?? 'Unknown User') ?></div>
                                        <div class="user-email"><?= htmlspecialchars($transaction['email'] ?? '') ?></div>
                                    </div>
                                </td>
                                <td>
                                    <?php if ($transaction['merchant_name']): ?>
                                        <div class="merchant-name"><?= htmlspecialchars($transaction['merchant_name']) ?></div>
                                        <div class="merchant-id"><?= htmlspecialchars($transaction['merchant_id']) ?></div>
                                    <?php else: ?>
                                        <span style="color: var(--text-light);">System</span>
                                    <?php endif; ?>
                                </td>
                                <td>
                                    <span class="<?= $transaction['amount'] >= 0 ? 'amount-positive' : 'amount-negative' ?>">
                                        <?= $transaction['amount'] >= 0 ? '+' : '' ?>RM <?= number_format(abs($transaction['amount']), 2) ?>
                                    </span>
                                </td>
                                <td>
                                    <span style="text-transform: capitalize;">
                                        <?= htmlspecialchars($transaction['transaction_type']) ?>
                                    </span>
                                </td>
                                <td>
                                    <span class="status-badge <?= $transaction['status'] ?>">
                                        <?= $transaction['status'] ?>
                                    </span>
                                </td>
                                <td>
                                    <div class="transaction-date">
                                        <?= date('M j, Y', strtotime($transaction['timestamp'])) ?><br>
                                        <small><?= date('H:i:s', strtotime($transaction['timestamp'])) ?></small>
                                    </div>
                                </td>
                                <td>
                                    <div class="action-buttons">
                                        <button class="btn btn-info" title="View Details" 
                                                onclick="viewTransaction(<?= htmlspecialchars(json_encode($transaction)) ?>)">
                                            <i class="fas fa-eye"></i>
                                        </button>
                                        <?php if ($transaction['status'] === 'success' && $transaction['transaction_type'] === 'payment'): ?>
                                            <button class="btn btn-warning" title="Process Refund" 
                                                    onclick="processRefund(<?= $transaction['id'] ?>, '<?= htmlspecialchars($transaction['full_name']) ?>', <?= abs($transaction['amount']) ?>)">
                                                <i class="fas fa-undo"></i>
                                            </button>
                                        <?php endif; ?>
                                    </div>
                                </td>
                            </tr>
                        <?php endforeach; ?>
                    <?php endif; ?>
                </tbody>
            </table>

            <!-- Pagination -->
            <?php if ($total_pages > 1): ?>
                <div class="pagination">
                    <?php if ($page > 1): ?>
                        <a href="?page=<?= $page - 1 ?>&date_from=<?= urlencode($date_from) ?>&date_to=<?= urlencode($date_to) ?>&merchant=<?= urlencode($merchant_filter) ?>&status=<?= urlencode($status_filter) ?>&type=<?= urlencode($type_filter) ?>&search=<?= urlencode($search) ?>">
                            <i class="fas fa-chevron-left"></i> Previous
                        </a>
                    <?php endif; ?>

                    <?php for ($i = max(1, $page - 2); $i <= min($total_pages, $page + 2); $i++): ?>
                        <?php if ($i == $page): ?>
                            <span class="current"><?= $i ?></span>
                        <?php else: ?>
                            <a href="?page=<?= $i ?>&date_from=<?= urlencode($date_from) ?>&date_to=<?= urlencode($date_to) ?>&merchant=<?= urlencode($merchant_filter) ?>&status=<?= urlencode($status_filter) ?>&type=<?= urlencode($type_filter) ?>&search=<?= urlencode($search) ?>"><?= $i ?></a>
                        <?php endif; ?>
                    <?php endfor; ?>

                    <?php if ($page < $total_pages): ?>
                        <a href="?page=<?= $page + 1 ?>&date_from=<?= urlencode($date_from) ?>&date_to=<?= urlencode($date_to) ?>&merchant=<?= urlencode($merchant_filter) ?>&status=<?= urlencode($status_filter) ?>&type=<?= urlencode($type_filter) ?>&search=<?= urlencode($search) ?>">
                            Next <i class="fas fa-chevron-right"></i>
                        </a>
                    <?php endif; ?>
                </div>
            <?php endif; ?>
        </div>
    </main>

    <!-- Transaction Details Modal -->
    <div id="transactionModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 class="modal-title">Transaction Details</h3>
                <button class="modal-close" onclick="closeModal('transactionModal')">&times;</button>
            </div>
            <div class="modal-body" id="transactionModalBody">
                <!-- Transaction details will be loaded here -->
            </div>
            <div class="modal-footer">
                <button type="button" class="btn-cancel" onclick="closeModal('transactionModal')">Close</button>
            </div>
        </div>
    </div>

    <!-- SIMPLIFIED Refund Modal -->
    <div id="refundModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 class="modal-title">Process Refund</h3>
                <button class="modal-close" onclick="closeModal('refundModal')">&times;</button>
            </div>
            <form method="POST" id="refundForm">
                <input type="hidden" name="action" value="process_refund">
                <input type="hidden" name="transaction_id" id="refund_transaction_id">
                <input type="hidden" name="refund_reason" value="Admin processed refund">
                <div class="modal-body">
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <strong>Transaction:</strong> #<span id="refund_transaction_display"></span><br>
                        <strong>User:</strong> <span id="refund_user_name"></span><br>
                        <strong>Refund Amount:</strong> RM <span id="refund_amount_display"></span>
                    </div>

                    <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 8px; font-size: 14px;">
                        <strong>⚠️ Warning:</strong> This action will immediately refund the full amount to the user's wallet and mark the original transaction as refunded. This action cannot be undone.
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn-cancel" onclick="closeModal('refundModal')">Cancel</button>
                    <button type="submit" class="btn-save" style="background: var(--warning);">Process Refund</button>
                </div>
            </form>
        </div>
    </div>

    <!-- SIMPLIFIED Export Modal for Merchant Reports -->
    <div id="exportModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 class="modal-title">Generate Merchant Report</h3>
                <button class="modal-close" onclick="closeModal('exportModal')">&times;</button>
            </div>
            <form action="export_merchant_report.php" method="POST" target="_blank" id="exportForm">
                <div class="modal-body">
                    <div class="form-group" style="margin-bottom: 20px;">
                        <label class="form-label">Select Merchant</label>
                        <select class="form-input" name="export_merchant_id" required>
                            <option value="">Choose a merchant...</option>
                            <?php foreach ($merchants as $merchant): ?>
                                <option value="<?= htmlspecialchars($merchant['merchant_id']) ?>">
                                    <?= htmlspecialchars($merchant['merchant_name']) ?> (<?= htmlspecialchars($merchant['merchant_id']) ?>)
                                </option>
                            <?php endforeach; ?>
                        </select>
                    </div>

                    <div class="form-group" style="margin-bottom: 20px;">
                        <label class="form-label">Report Period</label>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                            <input type="date" class="form-input" name="export_date_from" 
                                   value="<?= date('Y-m-d', strtotime('-30 days')) ?>" required>
                            <input type="date" class="form-input" name="export_date_to" 
                                   value="<?= date('Y-m-d') ?>" required>
                        </div>
                    </div>

                    <div style="background: #e3f2fd; border: 1px solid #2196f3; padding: 15px; border-radius: 8px; font-size: 14px;">
                        <strong><i class="fas fa-info-circle"></i> Report Contents:</strong><br>
                        • Total Revenue & Transaction Count<br>
                        • Success/Failure Rate Analysis<br>
                        • Daily Performance Trends<br>
                        • Detailed Transaction Log<br>
                        • Customer Purchase History
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn-cancel" onclick="closeModal('exportModal')">Cancel</button>
                    <button type="submit" class="btn-save" style="background: var(--success);">Generate Report</button>
                </div>
            </form>
        </div>
    </div>

    <script>
        // Chart.js configuration for revenue chart
        const ctx = document.getElementById('revenueChart').getContext('2d');
        const chartData = <?= json_encode($chart_data) ?>;
        
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartData.map(item => {
                    const date = new Date(item.date);
                    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                }),
                datasets: [{
                    label: 'Revenue (RM)',
                    data: chartData.map(item => parseFloat(item.revenue)),
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }, {
                    label: 'Transactions',
                    data: chartData.map(item => parseInt(item.transactions)),
                    borderColor: '#38a169',
                    backgroundColor: 'rgba(56, 161, 105, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Revenue (RM)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Transaction Count'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });

        // Modal functions
        function closeModal(modalId) {
            document.getElementById(modalId).classList.remove('show');
        }

        function viewTransaction(transaction) {
            const modalBody = document.getElementById('transactionModalBody');
            
            const timestamp = new Date(transaction.timestamp).toLocaleString();
            
            modalBody.innerHTML = `
                <div style="display: grid; gap: 20px;">
                    <div>
                        <h4 style="margin-bottom: 15px; color: var(--primary);">Transaction Information</h4>
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; line-height: 1.8;">
                            <strong>Transaction ID:</strong> #${transaction.id}<br>
                            <strong>Order ID:</strong> ${transaction.order_id || 'N/A'}<br>
                            <strong>Type:</strong> ${transaction.transaction_type}<br>
                            <strong>Method:</strong> ${transaction.payment_method}<br>
                            <strong>Status:</strong> <span class="status-badge ${transaction.status}">${transaction.status}</span>
                        </div>
                    </div>

                    <div>
                        <h4 style="margin-bottom: 15px; color: var(--success);">Amount Details</h4>
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; line-height: 1.8;">
                            <strong>Amount:</strong> RM ${Math.abs(transaction.amount).toFixed(2)}<br>
                            <strong>Currency:</strong> ${transaction.currency || 'MYR'}
                        </div>
                    </div>

                    <div>
                        <h4 style="margin-bottom: 15px; color: var(--info);">Parties Involved</h4>
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; line-height: 1.8;">
                            <strong>User:</strong> ${transaction.full_name || 'Unknown'}<br>
                            <strong>Email:</strong> ${transaction.email || 'N/A'}<br>
                            <strong>Merchant:</strong> ${transaction.merchant_name || 'System'}<br>
                            <strong>Merchant ID:</strong> ${transaction.merchant_id || 'N/A'}
                        </div>
                    </div>

                    <div>
                        <h4 style="margin-bottom: 15px; color: var(--warning);">Timing</h4>
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; line-height: 1.8;">
                            <strong>Created:</strong> ${timestamp}<br>
                        </div>
                    </div>
                </div>
            `;
            
            document.getElementById('transactionModal').classList.add('show');
        }

        function processRefund(transactionId, userName, amount) {
            document.getElementById('refund_transaction_id').value = transactionId;
            document.getElementById('refund_transaction_display').textContent = transactionId;
            document.getElementById('refund_user_name').textContent = userName;
            document.getElementById('refund_amount_display').textContent = amount.toFixed(2);
            
            document.getElementById('refundModal').classList.add('show');
        }

        function exportMerchantReport() {
            document.getElementById('exportModal').classList.add('show');
        }

        // Close modals when clicking outside
        document.addEventListener('click', function(e) {
            if (e.target.classList.contains('modal')) {
                e.target.classList.remove('show');
            }
        });

        // Auto-hide messages after 5 seconds
        document.addEventListener('DOMContentLoaded', function() {
            const messages = document.querySelectorAll('.message');
            messages.forEach(message => {
                setTimeout(() => {
                    message.style.opacity = '0';
                    message.style.transform = 'translateY(-10px)';
                    setTimeout(() => message.remove(), 300);
                }, 5000);
            });
        });

        // Form validation for refund
        document.getElementById('refundForm').addEventListener('submit', function(e) {
            if (!confirm('Are you sure you want to process this refund? This action cannot be undone.')) {
                e.preventDefault();
            }
        });

        console.log('Transaction Management System loaded successfully (Fixed Security Logging)');
    </script>
</body>
</html>