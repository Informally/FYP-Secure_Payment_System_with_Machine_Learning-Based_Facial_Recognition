<?php
// transaction_history.php - Complete Transaction History with Filters
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

// Pagination settings
$items_per_page = 5; // Show only 5 transactions per page
$current_page = isset($_GET['page']) ? max(1, intval($_GET['page'])) : 1;
$offset = ($current_page - 1) * $items_per_page;

// Filter settings
$type_filter = $_GET['type'] ?? 'all';
$date_filter = $_GET['date'] ?? 'all';
$search_filter = $_GET['search'] ?? '';

try {
    $conn = dbConnect();
    
    // Get user details for header
    $stmt = $conn->prepare("SELECT full_name FROM users WHERE id = ?");
    $stmt->bind_param("i", $user_db_id);
    $stmt->execute();
    $result = $stmt->get_result();
    $user = $result->fetch_assoc();
    $stmt->close();
    
    // Build WHERE clause for filters
    $where_conditions = ["user_id = ?"];
    $params = [$user_db_id];
    $param_types = "i";
    
    if ($type_filter !== 'all') {
        $where_conditions[] = "transaction_type = ?";
        $params[] = $type_filter;
        $param_types .= "s";
    }
    
    if ($date_filter !== 'all') {
        switch ($date_filter) {
            case 'today':
                $where_conditions[] = "DATE(timestamp) = CURDATE()";
                break;
            case 'week':
                $where_conditions[] = "YEARWEEK(timestamp) = YEARWEEK(CURDATE())";
                break;
            case 'month':
                $where_conditions[] = "YEAR(timestamp) = YEAR(CURDATE()) AND MONTH(timestamp) = MONTH(CURDATE())";
                break;
            case 'year':
                $where_conditions[] = "YEAR(timestamp) = YEAR(CURDATE())";
                break;
        }
    }
    
    if (!empty($search_filter)) {
        $where_conditions[] = "(transaction_id LIKE ? OR merchant_name LIKE ? OR order_id LIKE ?)";
        $search_term = "%{$search_filter}%";
        $params[] = $search_term;
        $params[] = $search_term;
        $params[] = $search_term;
        $param_types .= "sss";
    }
    
    $where_clause = implode(" AND ", $where_conditions);
    
    // Get total count for pagination
    $count_query = "SELECT COUNT(*) as total FROM transactions WHERE {$where_clause}";
    $stmt = $conn->prepare($count_query);
    if (!empty($params)) {
        $stmt->bind_param($param_types, ...$params);
    }
    $stmt->execute();
    $result = $stmt->get_result();
    $total_records = $result->fetch_assoc()['total'];
    $total_pages = ceil($total_records / $items_per_page);
    $stmt->close();
    
    // Get transactions with pagination
    $query = "
        SELECT 
            id, transaction_id, amount, status, payment_method, timestamp,
            order_id, merchant_id, merchant_name, transaction_type, currency,
            recipient_id, failure_reason, ip_address
        FROM transactions 
        WHERE {$where_clause}
        ORDER BY timestamp DESC 
        LIMIT ? OFFSET ?
    ";
    
    $stmt = $conn->prepare($query);
    $params[] = $items_per_page;
    $params[] = $offset;
    $param_types .= "ii";
    $stmt->bind_param($param_types, ...$params);
    $stmt->execute();
    $result = $stmt->get_result();
    
    $transactions = [];
    while ($row = $result->fetch_assoc()) {
        $transactions[] = $row;
    }
    $stmt->close();
    
    // Get summary statistics
    $summary_query = "
        SELECT 
            COUNT(*) as total_transactions,
            SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) as total_received,
            SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) as total_spent,
            COUNT(CASE WHEN status = 'success' THEN 1 END) as successful_transactions
        FROM transactions 
        WHERE {$where_clause}
    ";
    
    $stmt = $conn->prepare($summary_query);
    if (count($params) > 2) { // Remove the LIMIT and OFFSET params for summary
        $summary_params = array_slice($params, 0, -2);
        $summary_types = substr($param_types, 0, -2);
        $stmt->bind_param($summary_types, ...$summary_params);
    }
    $stmt->execute();
    $result = $stmt->get_result();
    $summary = $result->fetch_assoc();
    $stmt->close();
    
    $conn->close();
    
} catch (Exception $e) {
    error_log("Transaction history error: " . $e->getMessage());
    $error = "Unable to load transaction history.";
    $transactions = [];
    $summary = [
        'total_transactions' => 0,
        'total_received' => 0,
        'total_spent' => 0,
        'successful_transactions' => 0
    ];
    $total_pages = 1;
    $total_records = 0;
}

function getStatusBadgeClass($status) {
    switch ($status) {
        case 'success': return 'status-success';
        case 'failed': return 'status-failed';
        case 'pending': return 'status-pending';
        default: return 'status-pending';
    }
}

function getTransactionIcon($type, $amount) {
    if ($amount > 0) {
        return 'fas fa-plus-circle';
    } else {
        switch ($type) {
            case 'payment': return 'fas fa-store';
            case 'transfer_send': return 'fas fa-paper-plane';
            case 'transfer_receive': return 'fas fa-hand-holding-usd';
            default: return 'fas fa-exchange-alt';
        }
    }
}

function getTransactionColor($amount) {
    return $amount > 0 ? 'transaction-incoming' : 'transaction-outgoing';
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transaction History - Secure FacePay</title>
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
        
        /* Main Content */
        .main-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        
        /* Summary Cards */
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .summary-card {
            background: white;
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            text-align: center;
        }
        
        .summary-value {
            font-size: 28px;
            font-weight: 700;
            margin: 10px 0;
        }
        
        .summary-label {
            color: var(--text-light);
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .value-received {
            color: var(--success);
        }
        
        .value-spent {
            color: var(--error);
        }
        
        .value-primary {
            color: var(--primary);
        }
        
        .value-warning {
            color: var(--warning);
        }
        
        /* Filters Section */
        .filters-section {
            background: white;
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        }
        
        .filters-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .filters-title {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-dark);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .clear-filters {
            background: var(--border);
            color: var(--text-medium);
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }
        
        .clear-filters:hover {
            background: var(--text-light);
            color: white;
        }
        
        .filters-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .filter-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .filter-label {
            font-weight: 600;
            color: var(--text-dark);
            font-size: 14px;
        }
        
        .filter-select, .filter-input {
            padding: 12px;
            border: 2px solid var(--border);
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        
        .filter-select:focus, .filter-input:focus {
            border-color: var(--primary);
            outline: none;
        }
        
        .search-container {
            position: relative;
            grid-column: span 2;
        }
        
        .search-input {
            width: 100%;
            padding: 12px 12px 12px 40px;
            border: 2px solid var(--border);
            border-radius: 8px;
            font-size: 14px;
        }
        
        .search-icon {
            position: absolute;
            left: 12px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-light);
        }
        
        /* Transactions Table */
        .transactions-section {
            background: white;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            overflow: hidden;
        }
        
        .transactions-header {
            padding: 25px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .transactions-title {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-dark);
        }
        
        .results-count {
            color: var(--text-light);
            font-size: 14px;
        }
        
        .transactions-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .transactions-table th,
        .transactions-table td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        
        .transactions-table th {
            background: var(--bg-light);
            font-weight: 600;
            color: var(--text-dark);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .transaction-row {
            transition: background-color 0.3s;
        }
        
        .transaction-row:hover {
            background-color: var(--bg-light);
        }
        
        .transaction-info {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .transaction-icon {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
        }
        
        .transaction-incoming {
            background: rgba(56, 161, 105, 0.1);
            color: var(--success);
        }
        
        .transaction-outgoing {
            background: rgba(237, 137, 54, 0.1);
            color: var(--warning);
        }
        
        .transaction-details h4 {
            font-weight: 600;
            color: var(--text-dark);
            margin-bottom: 2px;
        }
        
        .transaction-id {
            font-size: 12px;
            color: var(--text-light);
            font-family: monospace;
        }
        
        .amount-cell {
            text-align: right;
            font-weight: 600;
        }
        
        .amount-positive {
            color: var(--success);
        }
        
        .amount-negative {
            color: var(--error);
        }
        
        .status-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .status-success {
            background: rgba(56, 161, 105, 0.1);
            color: var(--success);
        }
        
        .status-failed {
            background: rgba(229, 62, 62, 0.1);
            color: var(--error);
        }
        
        .status-pending {
            background: rgba(237, 137, 54, 0.1);
            color: var(--warning);
        }
        
        .merchant-info {
            font-size: 14px;
            color: var(--text-medium);
        }
        
        .date-info {
            font-size: 14px;
            color: var(--text-medium);
        }
        
        /* Pagination */
        .pagination-section {
            padding: 25px;
            border-top: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .pagination-info {
            color: var(--text-light);
            font-size: 14px;
        }
        
        .pagination-controls {
            display: flex;
            gap: 10px;
        }
        
        .pagination-btn {
            padding: 8px 12px;
            border: 1px solid var(--border);
            background: white;
            color: var(--text-medium);
            text-decoration: none;
            border-radius: 6px;
            font-size: 14px;
            transition: all 0.3s;
        }
        
        .pagination-btn:hover {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
            text-decoration: none;
        }
        
        .pagination-btn.active {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }
        
        .pagination-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-light);
        }
        
        .empty-icon {
            width: 80px;
            height: 80px;
            background: var(--border);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px;
            font-size: 32px;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .main-content {
                padding: 20px 15px;
            }
            
            .summary-grid {
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
            }
            
            .filters-grid {
                grid-template-columns: 1fr;
            }
            
            .search-container {
                grid-column: span 1;
            }
            
            .transactions-table {
                font-size: 14px;
            }
            
            .transactions-table th,
            .transactions-table td {
                padding: 10px 8px;
            }
            
            .transaction-info {
                flex-direction: column;
                align-items: flex-start;
                gap: 8px;
            }
            
            .pagination-section {
                flex-direction: column;
                gap: 15px;
            }
        }
        
        @media (max-width: 480px) {
            .summary-grid {
                grid-template-columns: 1fr;
            }
            
            .header-content {
                flex-direction: column;
                gap: 15px;
                text-align: center;
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
                <i class="fas fa-history"></i>
                Transaction History
            </div>
            <div style="width: 150px;"></div> <!-- Spacer for centering -->
        </div>
    </header>

    <!-- Main Content -->
    <main class="main-content">
        <?php if (isset($error)): ?>
            <div style="background: var(--error); color: white; padding: 20px; border-radius: 12px; margin-bottom: 20px;">
                <i class="fas fa-exclamation-triangle"></i>
                <?= htmlspecialchars($error) ?>
            </div>
        <?php endif; ?>

        <!-- Summary Statistics -->
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-value value-primary"><?= number_format($summary['total_transactions']) ?></div>
                <div class="summary-label">Total Transactions</div>
            </div>
            <div class="summary-card">
                <div class="summary-value value-received">RM <?= number_format($summary['total_received'], 2) ?></div>
                <div class="summary-label">Total Received</div>
            </div>
            <div class="summary-card">
                <div class="summary-value value-spent">RM <?= number_format($summary['total_spent'], 2) ?></div>
                <div class="summary-label">Total Spent</div>
            </div>
            <div class="summary-card">
                <div class="summary-value value-warning"><?= number_format($summary['successful_transactions']) ?></div>
                <div class="summary-label">Successful</div>
            </div>
        </div>

        <!-- Filters Section -->
        <div class="filters-section">
            <div class="filters-header">
                <div class="filters-title">
                    <i class="fas fa-filter"></i>
                    Filter Transactions
                </div>
                <button class="clear-filters" onclick="clearAllFilters()">
                    <i class="fas fa-times"></i> Clear All
                </button>
            </div>
            
            <form method="GET" id="filtersForm">
                <div class="filters-grid">
                    <div class="filter-group">
                        <label class="filter-label">Type</label>
                        <select name="type" class="filter-select" onchange="this.form.submit()">
                            <option value="all" <?= $type_filter === 'all' ? 'selected' : '' ?>>All Types</option>
                            <option value="payment" <?= $type_filter === 'payment' ? 'selected' : '' ?>>Payment</option>
                            <option value="topup" <?= $type_filter === 'topup' ? 'selected' : '' ?>>Top Up</option>
                        </select>
                    </div>
                    
                    <div class="filter-group">
                        <label class="filter-label">Date Range</label>
                        <select name="date" class="filter-select" onchange="this.form.submit()">
                            <option value="all" <?= $date_filter === 'all' ? 'selected' : '' ?>>All Time</option>
                            <option value="today" <?= $date_filter === 'today' ? 'selected' : '' ?>>Today</option>
                            <option value="week" <?= $date_filter === 'week' ? 'selected' : '' ?>>This Week</option>
                            <option value="month" <?= $date_filter === 'month' ? 'selected' : '' ?>>This Month</option>
                            <option value="year" <?= $date_filter === 'year' ? 'selected' : '' ?>>This Year</option>
                        </select>
                    </div>
                    
                    <div class="search-container">
                        <label class="filter-label">Search</label>
                        <div style="position: relative;">
                            <i class="fas fa-search search-icon"></i>
                            <input type="text" name="search" class="search-input" 
                                   placeholder="Search by Transaction ID, Merchant, or Order ID..." 
                                   value="<?= htmlspecialchars($search_filter) ?>">
                        </div>
                    </div>
                </div>
            </form>
        </div>

        <!-- Transactions Table -->
        <div class="transactions-section">
            <div class="transactions-header">
                <div class="transactions-title">Transaction History</div>
                <div class="results-count">
                    Showing <?= count($transactions) ?> of <?= number_format($total_records) ?> transactions
                </div>
            </div>

            <?php if (count($transactions) > 0): ?>
                <table class="transactions-table">
                    <thead>
                        <tr>
                            <th>Transaction</th>
                            <th>Amount</th>
                            <th>Status</th>
                            <th>Merchant</th>
                            <th>Date</th>
                            <th>Method</th>
                        </tr>
                    </thead>
                    <tbody>
                        <?php foreach ($transactions as $transaction): ?>
                            <tr class="transaction-row">
                                <td>
                                    <div class="transaction-info">
                                        <div class="transaction-icon <?= getTransactionColor($transaction['amount']) ?>">
                                            <i class="<?= getTransactionIcon($transaction['transaction_type'], $transaction['amount']) ?>"></i>
                                        </div>
                                        <div class="transaction-details">
                                            <h4><?= ucwords(str_replace('_', ' ', $transaction['transaction_type'])) ?></h4>
                                            <div class="transaction-id"><?= htmlspecialchars($transaction['transaction_id']) ?></div>
                                        </div>
                                    </div>
                                </td>
                                <td class="amount-cell">
                                    <span class="<?= $transaction['amount'] > 0 ? 'amount-positive' : 'amount-negative' ?>">
                                        <?= $transaction['amount'] > 0 ? '+' : '' ?>RM <?= number_format(abs($transaction['amount']), 2) ?>
                                    </span>
                                </td>
                                <td>
                                    <span class="status-badge <?= getStatusBadgeClass($transaction['status']) ?>">
                                        <?= ucfirst($transaction['status']) ?>
                                    </span>
                                </td>
                                <td>
                                    <div class="merchant-info">
                                        <?= $transaction['merchant_name'] ? htmlspecialchars($transaction['merchant_name']) : 'N/A' ?>
                                        <?php if ($transaction['order_id']): ?>
                                            <br><small style="color: var(--text-light);">Order: <?= htmlspecialchars($transaction['order_id']) ?></small>
                                        <?php endif; ?>
                                    </div>
                                </td>
                                <td>
                                    <div class="date-info">
                                        <?= date('M j, Y', strtotime($transaction['timestamp'])) ?>
                                        <br><small style="color: var(--text-light);"><?= date('g:i A', strtotime($transaction['timestamp'])) ?></small>
                                    </div>
                                </td>
                                <td>
                                    <span style="text-transform: capitalize;">
                                        <?= htmlspecialchars($transaction['payment_method']) ?>
                                    </span>
                                </td>
                            </tr>
                        <?php endforeach; ?>
                    </tbody>
                </table>

                <!-- Pagination -->
                <?php if ($total_pages > 1): ?>
                    <div class="pagination-section">
                        <div class="pagination-info">
                            Page <?= $current_page ?> of <?= $total_pages ?>
                        </div>
                        <div class="pagination-controls">
                            <!-- <?php if ($current_page > 1): ?>
                                <a href="?<?= http_build_query(array_merge($_GET, ['page' => 1])) ?>" class="pagination-btn">First</a>
                                <a href="?<?= http_build_query(array_merge($_GET, ['page' => $current_page - 1])) ?>" class="pagination-btn">Previous</a>
                            <?php endif; ?> -->
                            
                            <?php
                            $start_page = max(1, $current_page - 2);
                            $end_page = min($total_pages, $current_page + 2);
                            for ($i = $start_page; $i <= $end_page; $i++):
                            ?>
                                <a href="?<?= http_build_query(array_merge($_GET, ['page' => $i])) ?>" 
                                   class="pagination-btn <?= $i === $current_page ? 'active' : '' ?>">
                                    <?= $i ?>
                                </a>
                            <?php endfor; ?>
                            
                            <!-- <?php if ($current_page < $total_pages): ?>
                                <a href="?<?= http_build_query(array_merge($_GET, ['page' => $current_page + 1])) ?>" class="pagination-btn">Next</a>
                                <a href="?<?= http_build_query(array_merge($_GET, ['page' => $total_pages])) ?>" class="pagination-btn">Last</a>
                            <?php endif; ?> -->
                        </div>
                    </div>
                <?php endif; ?>

            <?php else: ?>
                <!-- Empty State -->
                <div class="empty-state">
                    <div class="empty-icon">
                        <i class="fas fa-receipt"></i>
                    </div>
                    <h3 style="margin-bottom: 10px;">No Transactions Found</h3>
                    <p>
                        <?php if ($type_filter !== 'all' || $date_filter !== 'all' || !empty($search_filter)): ?>
                            No transactions match your current filters. Try adjusting your search criteria.
                        <?php else: ?>
                            You haven't made any transactions yet. <a href="topup.php" style="color: var(--primary);">Top up your wallet</a> to get started!
                        <?php endif; ?>
                    </p>
                    <?php if ($type_filter !== 'all' || $date_filter !== 'all' || !empty($search_filter)): ?>
                        <button onclick="clearAllFilters()" style="margin-top: 15px; padding: 10px 20px; background: var(--primary); color: white; border: none; border-radius: 8px; cursor: pointer;">
                            Clear All Filters
                        </button>
                    <?php endif; ?>
                </div>
            <?php endif; ?>
        </div>
    </main>

    <script>
        // Clear all filters
        function clearAllFilters() {
            window.location.href = 'transaction_history.php';
        }

        // Auto-submit search after typing delay
        let searchTimeout;
        const searchInput = document.querySelector('input[name="search"]');
        if (searchInput) {
            searchInput.addEventListener('input', function() {
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(() => {
                    this.form.submit();
                }, 1000); // Submit after 1 second of no typing
            });
        }

        // Add loading states to filter changes
        document.querySelectorAll('.filter-select').forEach(select => {
            select.addEventListener('change', function() {
                this.style.opacity = '0.6';
                this.disabled = true;
                
                // Re-enable after form submission
                setTimeout(() => {
                    this.style.opacity = '1';
                    this.disabled = false;
                }, 100);
            });
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            // Ctrl/Cmd + K to focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                searchInput.focus();
            }
            
            // Escape to clear search
            if (e.key === 'Escape' && document.activeElement === searchInput) {
                searchInput.value = '';
                searchInput.form.submit();
            }
        });

        // Add hover effects to transaction rows
        document.querySelectorAll('.transaction-row').forEach(row => {
            row.addEventListener('mouseenter', function() {
                this.style.transform = 'translateX(5px)';
            });
            
            row.addEventListener('mouseleave', function() {
                this.style.transform = 'translateX(0)';
            });
        });

        // Smooth scrolling for pagination
        document.querySelectorAll('.pagination-btn').forEach(btn => {
            btn.addEventListener('click', function(e) {
                // Add loading state
                this.style.opacity = '0.6';
                this.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            });
        });

        // Auto-refresh data every 30 seconds if on first page
        <?php if ($current_page === 1): ?>
        setInterval(() => {
            // Only refresh if no filters are applied and user hasn't interacted recently
            const lastInteraction = sessionStorage.getItem('lastInteraction');
            const now = Date.now();
            
            if (!lastInteraction || (now - parseInt(lastInteraction)) > 30000) {
                // Check if current URL has no filters
                const urlParams = new URLSearchParams(window.location.search);
                const hasFilters = urlParams.has('type') || 
                                 urlParams.has('date') || urlParams.has('search');
                
                if (!hasFilters) {
                    window.location.reload();
                }
            }
        }, 30000); // 30 seconds

        // Track user interactions
        ['click', 'keypress', 'scroll'].forEach(event => {
            document.addEventListener(event, () => {
                sessionStorage.setItem('lastInteraction', Date.now().toString());
            });
        });
        <?php endif; ?>

        // Add tooltips to status badges
        document.querySelectorAll('.status-badge').forEach(badge => {
            const status = badge.textContent.toLowerCase();
            let tooltip = '';
            
            switch (status) {
                case 'success':
                    tooltip = 'Transaction completed successfully';
                    break;
                case 'failed':
                    tooltip = 'Transaction failed to process';
                    break;
                case 'pending':
                    tooltip = 'Transaction is being processed';
                    break;
            }
            
            if (tooltip) {
                badge.title = tooltip;
            }
        });

        // Export functionality (placeholder)
        function exportTransactions() {
            alert('Export functionality coming soon!');
        }

        // Add keyboard navigation for pagination
        document.addEventListener('keydown', function(e) {
            if (e.target.tagName === 'INPUT') return; // Don't interfere with form inputs
            
            // Left arrow for previous page
            if (e.key === 'ArrowLeft' && <?= $current_page ?> > 1) {
                window.location.href = '?<?= http_build_query(array_merge($_GET, ['page' => $current_page - 1])) ?>';
            }
            
            // Right arrow for next page
            if (e.key === 'ArrowRight' && <?= $current_page ?> < <?= $total_pages ?>) {
                window.location.href = '?<?= http_build_query(array_merge($_GET, ['page' => $current_page + 1])) ?>';
            }
        });

        // Add animation to summary cards
        document.addEventListener('DOMContentLoaded', function() {
            const summaryCards = document.querySelectorAll('.summary-card');
            summaryCards.forEach((card, index) => {
                card.style.opacity = '0';
                card.style.transform = 'translateY(20px)';
                
                setTimeout(() => {
                    card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                    card.style.opacity = '1';
                    card.style.transform = 'translateY(0)';
                }, index * 100);
            });
        });
    </script>
</body>
</html>