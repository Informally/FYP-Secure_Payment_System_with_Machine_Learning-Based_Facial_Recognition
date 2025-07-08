<?php
// admin/dashboard.php
require_once 'config/admin_config.php';

// Require authentication
$admin = requireAdminAuth();

// Get dashboard data
$conn = dbConnect();

// Get summary statistics - FIXED for your 3 core features
$summary_query = "
    SELECT 
        (SELECT COUNT(*) FROM users WHERE account_status = 'active') as total_users,
        (SELECT COUNT(*) FROM merchants WHERE is_active = 1) as active_merchants,
        (SELECT COUNT(*) FROM transactions WHERE DATE(timestamp) = CURDATE()) as today_transactions,
        (SELECT COALESCE(SUM(ABS(amount)), 0) FROM transactions WHERE DATE(timestamp) = CURDATE() AND status = 'success' AND transaction_type = 'payment') as today_revenue,
        (SELECT COUNT(*) FROM system_alerts WHERE is_resolved = FALSE) as unresolved_alerts
";
$summary = $conn->query($summary_query)->fetch_assoc();

// Get recent transactions with merchant info
$stmt = $conn->prepare("
    SELECT t.*, u.full_name, m.name as merchant_name 
    FROM transactions t 
    LEFT JOIN users u ON t.user_id = u.id
    LEFT JOIN merchants m ON t.merchant_id = m.merchant_id
    WHERE t.transaction_type IN ('payment', 'Welcome Bonus')
    ORDER BY t.timestamp DESC 
    LIMIT 5
");
$stmt->execute();
$recent_transactions = $stmt->get_result()->fetch_all(MYSQLI_ASSOC);

// Get active alerts (system notifications)
$stmt = $conn->prepare("
    SELECT * FROM system_alerts 
    WHERE is_resolved = FALSE 
    ORDER BY severity DESC, created_at DESC 
    LIMIT 5
");
$stmt->execute();
$alerts = $stmt->get_result()->fetch_all(MYSQLI_ASSOC);

// Get merchant performance data
$stmt = $conn->prepare("
    SELECT 
        m.merchant_id,
        m.name,
        m.is_active,
        COUNT(t.id) as transaction_count,
        COALESCE(SUM(CASE WHEN t.status = 'success' THEN ABS(t.amount) ELSE 0 END), 0) as total_revenue,
        MAX(t.timestamp) as last_transaction
    FROM merchants m
    LEFT JOIN transactions t ON m.merchant_id = t.merchant_id
    GROUP BY m.id
    ORDER BY total_revenue DESC
    LIMIT 5
");
$stmt->execute();
$top_merchants = $stmt->get_result()->fetch_all(MYSQLI_ASSOC);

// Get hourly transaction data for chart (today)
$stmt = $conn->prepare("
    SELECT 
        HOUR(timestamp) as hour,
        COUNT(*) as transaction_count,
        SUM(CASE WHEN status = 'success' THEN ABS(amount) ELSE 0 END) as revenue
    FROM transactions 
    WHERE DATE(timestamp) = CURDATE() AND transaction_type = 'payment'
    GROUP BY HOUR(timestamp)
    ORDER BY hour
");
$stmt->execute();
$hourly_data = $stmt->get_result()->fetch_all(MYSQLI_ASSOC);

// Get user registration trend (last 7 days)
$stmt = $conn->prepare("
    SELECT 
        DATE(created_at) as date,
        COUNT(*) as new_users
    FROM users 
    WHERE DATE(created_at) >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
    GROUP BY DATE(created_at)
    ORDER BY date ASC
");
$stmt->execute();
$user_growth_data = $stmt->get_result()->fetch_all(MYSQLI_ASSOC);

$conn->close();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Dashboard - FacePay System</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
            margin-bottom: 30px;
        }

        .page-title {
            font-size: 32px;
            font-weight: 700;
            color: var(--text-dark);
            margin-bottom: 8px;
        }

        .page-subtitle {
            color: var(--text-light);
            font-size: 16px;
        }

        /* Stats Cards */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }

        .stat-card {
            background: white;
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            border-left: 4px solid transparent;
            transition: transform 0.3s ease;
        }

        .stat-card:hover {
            transform: translateY(-2px);
        }

        .stat-card.primary { border-left-color: var(--primary); }
        .stat-card.success { border-left-color: var(--success); }
        .stat-card.warning { border-left-color: var(--warning); }
        .stat-card.error { border-left-color: var(--error); }
        .stat-card.info { border-left-color: var(--info); }

        .stat-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .stat-icon {
            width: 50px;
            height: 50px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            color: white;
        }

        .stat-icon.primary { background: var(--primary); }
        .stat-icon.success { background: var(--success); }
        .stat-icon.warning { background: var(--warning); }
        .stat-icon.error { background: var(--error); }
        .stat-icon.info { background: var(--info); }

        .stat-value {
            font-size: 32px;
            font-weight: 700;
            color: var(--text-dark);
            margin-bottom: 5px;
        }

        .stat-label {
            color: var(--text-light);
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Dashboard Grid */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }

        .card {
            background: white;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            overflow: hidden;
        }

        .card-header {
            padding: 20px 25px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .card-title {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-dark);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .card-content {
            padding: 25px;
        }

        /* Chart */
        .chart-container {
            position: relative;
            height: 300px;
        }

        /* Recent Transactions */
        .transaction-item {
            display: flex;
            align-items: center;
            padding: 15px 0;
            border-bottom: 1px solid var(--border);
        }

        .transaction-item:last-child {
            border-bottom: none;
        }

        .transaction-icon {
            width: 40px;
            height: 40px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 15px;
            font-size: 16px;
            color: white;
        }

        .transaction-details {
            flex: 1;
        }

        .transaction-user {
            font-weight: 600;
            color: var(--text-dark);
            font-size: 14px;
        }

        .transaction-info {
            font-size: 12px;
            color: var(--text-light);
        }

        .transaction-amount {
            text-align: right;
        }

        .amount {
            font-weight: 600;
            font-size: 14px;
        }

        .amount.success { color: var(--success); }
        .amount.error { color: var(--error); }

        .transaction-time {
            font-size: 11px;
            color: var(--text-light);
        }

        /* Alerts */
        .alert-item {
            display: flex;
            align-items: flex-start;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            border-left: 4px solid transparent;
        }

        .alert-item.high { 
            background: #fff5f5; 
            border-left-color: var(--error); 
        }

        .alert-item.medium { 
            background: #fffbf0; 
            border-left-color: var(--warning); 
        }

        .alert-item.low { 
            background: #f0f9ff; 
            border-left-color: var(--info); 
        }

        .alert-icon {
            margin-right: 12px;
            margin-top: 2px;
        }

        .alert-content {
            flex: 1;
        }

        .alert-title {
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 4px;
        }

        .alert-message {
            font-size: 12px;
            color: var(--text-light);
            line-height: 1.4;
        }

        .alert-time {
            font-size: 11px;
            color: var(--text-light);
            margin-top: 5px;
        }

        /* Merchant Performance */
        .merchant-grid {
            display: grid;
            gap: 15px;
        }

        .merchant-item {
            display: flex;
            align-items: center;
            padding: 15px;
            background: var(--bg-light);
            border-radius: 8px;
            border-left: 4px solid transparent;
        }

        .merchant-item.active { border-left-color: var(--success); }
        .merchant-item.inactive { border-left-color: var(--error); }

        .merchant-avatar {
            width: 40px;
            height: 40px;
            background: var(--primary);
            color: white;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            margin-right: 15px;
        }

        .merchant-info {
            flex: 1;
        }

        .merchant-name {
            font-weight: 600;
            color: var(--text-dark);
            font-size: 14px;
        }

        .merchant-stats {
            font-size: 12px;
            color: var(--text-light);
        }

        .merchant-revenue {
            text-align: right;
        }

        .revenue-amount {
            font-weight: 600;
            color: var(--success);
            font-size: 14px;
        }

        .revenue-transactions {
            font-size: 11px;
            color: var(--text-light);
        }

        /* Quick Actions */
        .quick-actions {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .action-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            transition: transform 0.3s ease;
            text-decoration: none;
            color: var(--text-dark);
        }

        .action-card:hover {
            transform: translateY(-2px);
            text-decoration: none;
            color: var(--text-dark);
        }

        .action-icon {
            width: 60px;
            height: 60px;
            margin: 0 auto 15px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            color: white;
        }

        .action-title {
            font-weight: 600;
            margin-bottom: 5px;
        }

        .action-description {
            font-size: 12px;
            color: var(--text-light);
        }

        /* Responsive */
        @media (max-width: 1200px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }

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

            .stats-grid {
                grid-template-columns: 1fr;
            }

            .quick-actions {
                grid-template-columns: 1fr;
            }
        }

        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
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
                <a href="dashboard.php" class="nav-item active">
                    <i class="fas fa-tachometer-alt"></i> Dashboard
                </a>
                <a href="merchants.php" class="nav-item">
                    <i class="fas fa-store"></i> Merchants
                </a>
                <a href="users.php" class="nav-item">
                    <i class="fas fa-users"></i> Users
                </a>
                <a href="transactions.php" class="nav-item">
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
                <a href="logout.php" class="nav-item" title="Logout">
                    <i class="fas fa-sign-out-alt"></i>
                </a>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="main-content">
        <div class="page-header">
            <h1 class="page-title">Dashboard Overview</h1>
            <p class="page-subtitle">Monitor your FacePay system performance and key metrics</p>
        </div>

        <!-- Quick Actions -->
        <!-- <div class="quick-actions">
            <a href="merchants.php" class="action-card">
                <div class="action-icon" style="background: var(--primary);">
                    <i class="fas fa-store"></i>
                </div>
                <div class="action-title">Manage Merchants</div>
                <div class="action-description">Add and configure payment partners</div>
            </a>
            <a href="users.php" class="action-card">
                <div class="action-icon" style="background: var(--success);">
                    <i class="fas fa-users"></i>
                </div>
                <div class="action-title">User Management</div>
                <div class="action-description">View and manage customer accounts</div>
            </a>
            <a href="transactions.php" class="action-card">
                <div class="action-icon" style="background: var(--info);">
                    <i class="fas fa-credit-card"></i>
                </div>
                <div class="action-title">Transaction History</div>
                <div class="action-description">Monitor payments and revenue</div>
            </a>
        </div> -->

        <!-- Stats Cards -->
        <div class="stats-grid">
            <div class="stat-card primary">
                <div class="stat-header">
                    <div class="stat-icon primary">
                        <i class="fas fa-users"></i>
                    </div>
                </div>
                <div class="stat-value"><?= number_format($summary['total_users'] ?? 0) ?></div>
                <div class="stat-label">Active Users</div>
            </div>

            <div class="stat-card success">
                <div class="stat-header">
                    <div class="stat-icon success">
                        <i class="fas fa-store"></i>
                    </div>
                </div>
                <div class="stat-value"><?= $summary['active_merchants'] ?? 0 ?></div>
                <div class="stat-label">Active Merchants</div>
            </div>

            <div class="stat-card info">
                <div class="stat-header">
                    <div class="stat-icon info">
                        <i class="fas fa-credit-card"></i>
                    </div>
                </div>
                <div class="stat-value"><?= number_format($summary['today_transactions'] ?? 0) ?></div>
                <div class="stat-label">Today's Transactions</div>
            </div>

            <div class="stat-card warning">
                <div class="stat-header">
                    <div class="stat-icon warning">
                        <i class="fas fa-dollar-sign"></i>
                    </div>
                </div>
                <div class="stat-value">RM <?= number_format($summary['today_revenue'] ?? 0, 2) ?></div>
                <div class="stat-label">Today's Revenue</div>
            </div>

            <?php if (($summary['unresolved_alerts'] ?? 0) > 0): ?>
            <div class="stat-card error">
                <div class="stat-header">
                    <div class="stat-icon error">
                        <i class="fas fa-exclamation-triangle"></i>
                    </div>
                </div>
                <div class="stat-value"><?= $summary['unresolved_alerts'] ?? 0 ?></div>
                <div class="stat-label">Active Alerts</div>
            </div>
            <?php endif; ?>
        </div>

        <!-- Dashboard Grid -->
        <div class="dashboard-grid">
            <!-- Transaction Chart -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fas fa-chart-line"></i>
                        Today's Transaction Activity
                    </div>
                </div>
                <div class="card-content">
                    <div class="chart-container">
                        <canvas id="transactionChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- Recent Activity -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fas fa-clock"></i>
                        Recent Transactions
                    </div>
                    <a href="transactions.php" style="color: var(--primary); text-decoration: none; font-size: 14px;">
                        View All <i class="fas fa-arrow-right"></i>
                    </a>
                </div>
                <div class="card-content">
                    <?php if (empty($recent_transactions)): ?>
                        <div style="text-align: center; padding: 40px; color: var(--text-light);">
                            <i class="fas fa-inbox" style="font-size: 48px; margin-bottom: 15px;"></i>
                            <p>No recent transactions</p>
                        </div>
                    <?php else: ?>
                        <?php foreach ($recent_transactions as $transaction): ?>
                            <div class="transaction-item">
                                <div class="transaction-icon <?= $transaction['status'] === 'success' ? 'success' : 'error' ?>" 
                                     style="background: <?= $transaction['status'] === 'success' ? 'var(--success)' : 'var(--error)' ?>;">
                                    <i class="fas fa-<?= $transaction['status'] === 'success' ? 'check' : 'times' ?>"></i>
                                </div>
                                <div class="transaction-details">
                                    <div class="transaction-user">
                                        <?= htmlspecialchars($transaction['full_name'] ?? 'Unknown User') ?>
                                    </div>
                                    <div class="transaction-info">
                                        via <?= htmlspecialchars($transaction['merchant_name'] ?? 'Direct Payment') ?>
                                    </div>
                                </div>
                                <div class="transaction-amount">
                                    <div class="amount <?= $transaction['status'] === 'success' ? 'success' : 'error' ?>">
                                        RM <?= number_format(abs($transaction['amount']), 2) ?>
                                    </div>
                                    <div class="transaction-time">
                                        <?= date('H:i', strtotime($transaction['timestamp'])) ?>
                                    </div>
                                </div>
                            </div>
                        <?php endforeach; ?>
                    <?php endif; ?>
                </div>
            </div>
        </div>

        <!-- Bottom Grid
        <div class="dashboard-grid">
            System Alerts
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fas fa-bell"></i>
                        System Alerts
                    </div>
                </div>
                <div class="card-content">
                    <?php if (empty($alerts)): ?>
                        <div style="text-align: center; padding: 40px; color: var(--text-light);">
                            <i class="fas fa-check-circle" style="font-size: 48px; margin-bottom: 15px; color: var(--success);"></i>
                            <p>All systems running normally</p>
                        </div>
                    <?php else: ?>
                        <?php foreach ($alerts as $alert): ?>
                            <div class="alert-item <?= htmlspecialchars($alert['severity']) ?>">
                                <div class="alert-icon">
                                    <i class="fas fa-<?= $alert['severity'] === 'high' ? 'exclamation-circle' : ($alert['severity'] === 'medium' ? 'exclamation-triangle' : 'info-circle') ?>"></i>
                                </div>
                                <div class="alert-content">
                                    <div class="alert-title"><?= htmlspecialchars($alert['title']) ?></div>
                                    <div class="alert-message"><?= htmlspecialchars($alert['message']) ?></div>
                                    <div class="alert-time"><?= date('M j, Y H:i', strtotime($alert['created_at'])) ?></div>
                                </div>
                            </div>
                        <?php endforeach; ?>
                    <?php endif; ?>
                </div>
            </div> -->

            <!-- Top Merchants -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fas fa-crown"></i>
                        Top Performing Merchants
                    </div>
                    <a href="merchants.php" style="color: var(--primary); text-decoration: none; font-size: 14px;">
                        Manage <i class="fas fa-cog"></i>
                    </a>
                </div>
                <div class="card-content">
                    <div class="merchant-grid">
                        <?php if (empty($top_merchants)): ?>
                            <div style="text-align: center; padding: 40px; color: var(--text-light);">
                                <i class="fas fa-store" style="font-size: 48px; margin-bottom: 15px;"></i>
                                <p>No merchants registered yet</p>
                            </div>
                        <?php else: ?>
                            <?php foreach ($top_merchants as $merchant): ?>
                                <div class="merchant-item <?= $merchant['is_active'] ? 'active' : 'inactive' ?>">
                                    <div class="merchant-avatar">
                                        <?= strtoupper(substr($merchant['name'], 0, 2)) ?>
                                    </div>
                                    <div class="merchant-info">
                                        <div class="merchant-name"><?= htmlspecialchars($merchant['name']) ?></div>
                                        <div class="merchant-stats">
                                            <?= $merchant['transaction_count'] ?> transactions
                                            <?php if ($merchant['last_transaction']): ?>
                                                • Last: <?= date('M j', strtotime($merchant['last_transaction'])) ?>
                                            <?php endif; ?>
                                        </div>
                                    </div>
                                    <div class="merchant-revenue">
                                        <div class="revenue-amount">RM <?= number_format($merchant['total_revenue'], 2) ?></div>
                                        <div class="revenue-transactions">Total Revenue</div>
                                    </div>
                                </div>
                            <?php endforeach; ?>
                        <?php endif; ?>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script>
        // Transaction Chart
        const ctx = document.getElementById('transactionChart').getContext('2d');
        
        // Prepare chart data
        const hourlyData = <?= json_encode($hourly_data) ?>;
        const hours = [];
        const transactions = [];
        const revenue = [];
        
        // Fill in missing hours with 0 values
        for (let i = 0; i < 24; i++) {
            hours.push(i + ':00');
            const hourData = hourlyData.find(d => parseInt(d.hour) === i);
            transactions.push(hourData ? parseInt(hourData.transaction_count) : 0);
            revenue.push(hourData ? parseFloat(hourData.revenue) : 0);
        }
        
        const transactionChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: hours,
                datasets: [{
                    label: 'Transactions',
                    data: transactions,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true,
                    yAxisID: 'y'
                }, {
                    label: 'Revenue (RM)',
                    data: revenue,
                    borderColor: '#38a169',
                    backgroundColor: 'rgba(56, 161, 105, 0.1)',
                    tension: 0.4,
                    fill: false,
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Hour of Day'
                        },
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Transactions'
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Revenue (RM)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                if (context.datasetIndex === 0) {
                                    return `Transactions: ${context.parsed.y}`;
                                } else {
                                    return `Revenue: RM ${context.parsed.y.toFixed(2)}`;
                                }
                            }
                        }
                    }
                }
            }
        });

        // Auto-refresh dashboard data every 30 seconds
        setInterval(function() {
            // Only refresh if page is visible
            if (!document.hidden) {
                fetch('api/dashboard-refresh.php')
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            updateDashboardStats(data.stats);
                            console.log('Dashboard data refreshed');
                        }
                    })
                    .catch(error => {
                        console.log('Dashboard refresh skipped - no API endpoint');
                    });
            }
        }, 30000);

        // Real-time updates functions
        function updateDashboardStats(stats) {
            // Update stat cards with new data
            const statCards = document.querySelectorAll('.stat-card');
            
            // Update total users
            const usersCard = statCards[0]?.querySelector('.stat-value');
            if (usersCard && stats.total_users) {
                usersCard.textContent = new Intl.NumberFormat().format(stats.total_users);
            }
            
            // Update active merchants
            const merchantsCard = statCards[1]?.querySelector('.stat-value');
            if (merchantsCard && stats.active_merchants !== undefined) {
                merchantsCard.textContent = stats.active_merchants;
            }
            
            // Update today's transactions
            const transactionsCard = statCards[2]?.querySelector('.stat-value');
            if (transactionsCard && stats.today_transactions !== undefined) {
                transactionsCard.textContent = new Intl.NumberFormat().format(stats.today_transactions);
            }
            
            // Update today's revenue
            const revenueCard = statCards[3]?.querySelector('.stat-value');
            if (revenueCard && stats.today_revenue !== undefined) {
                revenueCard.textContent = 'RM ' + new Intl.NumberFormat('en-MY', { 
                    minimumFractionDigits: 2 
                }).format(stats.today_revenue);
            }
        }

        // Page visibility handling
        document.addEventListener('visibilitychange', function() {
            if (!document.hidden) {
                console.log('Dashboard became visible, data will refresh on next interval');
            }
        });

        // Notification system for admin alerts
        function showNotification(message, type = 'info') {
            const notification = document.createElement('div');
            notification.className = `notification ${type}`;
            notification.textContent = message;
            
            Object.assign(notification.style, {
                position: 'fixed',
                top: '20px',
                right: '20px',
                padding: '15px 20px',
                borderRadius: '8px',
                color: 'white',
                backgroundColor: type === 'error' ? '#e53e3e' : type === 'success' ? '#38a169' : '#3182ce',
                zIndex: '1000',
                boxShadow: '0 4px 15px rgba(0,0,0,0.2)',
                transform: 'translateX(400px)',
                transition: 'transform 0.3s ease'
            });
            
            document.body.appendChild(notification);
            
            // Animate in
            setTimeout(() => {
                notification.style.transform = 'translateX(0)';
            }, 100);
            
            // Remove after 5 seconds
            setTimeout(() => {
                notification.style.transform = 'translateX(400px)';
                setTimeout(() => notification.remove(), 300);
            }, 5000);
        }

        // Quick navigation keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            // Alt + M for Merchants
            if (e.altKey && e.key === 'm') {
                e.preventDefault();
                window.location.href = 'merchants.php';
            }
            
            // Alt + U for Users
            if (e.altKey && e.key === 'u') {
                e.preventDefault();
                window.location.href = 'users.php';
            }
            
            // Alt + T for Transactions
            if (e.altKey && e.key === 't') {
                e.preventDefault();
                window.location.href = 'transactions.php';
            }
            
            // Alt + D for Dashboard
            if (e.altKey && e.key === 'd') {
                e.preventDefault();
                window.location.href = 'dashboard.php';
            }
        });

        // Welcome message and system info
        document.addEventListener('DOMContentLoaded', function() {
            const adminName = '<?= htmlspecialchars($admin['full_name']) ?>';
            const adminRole = '<?= htmlspecialchars($admin['role']) ?>';
            
            console.log(`🚀 FacePay Admin Dashboard loaded successfully!`);
            console.log(`👋 Welcome, ${adminName} (${adminRole})`);
            console.log(`⌨️  Keyboard shortcuts: Alt+M (Merchants), Alt+U (Users), Alt+T (Transactions), Alt+D (Dashboard)`);
            
            // Add subtle welcome notification for first-time users
            const lastVisit = localStorage.getItem('facepay_last_visit');
            const now = new Date().getTime();
            
            if (!lastVisit || now - parseInt(lastVisit) > 24 * 60 * 60 * 1000) {
                setTimeout(() => {
                    showNotification(`Welcome back, ${adminName}! 👋`, 'success');
                }, 1000);
                localStorage.setItem('facepay_last_visit', now.toString());
            }
        });

        // Enhanced error handling
        window.addEventListener('error', function(e) {
            console.error('Dashboard error:', e.error);
        });

        // Service worker registration for offline capability (optional)
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('sw.js').catch(function() {
                // Service worker not available, but that's okay
                console.log('Service worker not available - running online only');
            });
        }
    </script>
</body>
</html>