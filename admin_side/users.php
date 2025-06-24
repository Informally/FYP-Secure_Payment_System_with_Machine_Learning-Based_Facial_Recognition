<?php
require_once 'config/admin_config.php';

// Require authentication
$admin = requireAdminAuth(['super_admin', 'admin']);

$message = '';
$message_type = '';

// Handle form submissions for user actions
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $action = $_POST['action'] ?? '';
    
    try {
        $conn = dbConnect();
        $auth = getAdminAuth();
        
        if ($action === 'suspend_user') {
            $user_id = intval($_POST['user_id']);
            
            // Update user status to suspended
            $stmt = $conn->prepare("UPDATE users SET account_status = 'suspended' WHERE id = ?");
            $stmt->bind_param("i", $user_id);
            $stmt->execute();
            
            if ($stmt->affected_rows > 0) {
                // Log the action
                $auth->logAuditAction($admin['id'], 'suspend_user', 'users', $user_id, [
                    'status' => 'suspended'
                ]);
                
                $message = "User suspended successfully!";
                $message_type = 'success';
            } else {
                $message = "Failed to suspend user.";
                $message_type = 'error';
            }
            $stmt->close();
            
        } elseif ($action === 'activate_user') {
            $user_id = intval($_POST['user_id']);
            
            // Update user status to active
            $stmt = $conn->prepare("UPDATE users SET account_status = 'active' WHERE id = ?");
            $stmt->bind_param("i", $user_id);
            $stmt->execute();
            
            if ($stmt->affected_rows > 0) {
                // Log the action
                $auth->logAuditAction($admin['id'], 'activate_user', 'users', $user_id, [
                    'status' => 'active'
                ]);
                
                $message = "User activated successfully!";
                $message_type = 'success';
            } else {
                $message = "Failed to activate user.";
                $message_type = 'error';
            }
            $stmt->close();
        }
        
        $conn->close();
        
    } catch (Exception $e) {
        $message = "Error: " . $e->getMessage();
        $message_type = 'error';
    }
}

// Get search parameters
$search = $_GET['search'] ?? '';
$status_filter = $_GET['status'] ?? '';
$page = max(1, intval($_GET['page'] ?? 1));
$per_page = 20;
$offset = ($page - 1) * $per_page;

// Build query
$conn = dbConnect();
$where_conditions = [];
$params = [];
$param_types = '';

if (!empty($search)) {
    $where_conditions[] = "(u.full_name LIKE ? OR u.email LIKE ? OR u.user_id LIKE ?)";
    $search_param = "%{$search}%";
    $params[] = $search_param;
    $params[] = $search_param;
    $params[] = $search_param;
    $param_types .= 'sss';
}

if ($status_filter !== '') {
    $where_conditions[] = "u.account_status = ?";
    $params[] = $status_filter;
    $param_types .= 's';
}

$where_clause = !empty($where_conditions) ? 'WHERE ' . implode(' AND ', $where_conditions) : '';

// Get total count
$count_query = "
    SELECT COUNT(*) as total
    FROM users u
    {$where_clause}
";

if (!empty($params)) {
    $count_stmt = $conn->prepare($count_query);
    if (!empty($param_types)) {
        $count_stmt->bind_param($param_types, ...$params);
    }
    $count_stmt->execute();
    $total_users = $count_stmt->get_result()->fetch_assoc()['total'];
} else {
    $total_users = $conn->query($count_query)->fetch_assoc()['total'];
}

// Get users with wallet info and transaction data
$users_query = "
    SELECT 
        u.*,
        w.balance,
        COUNT(DISTINCT t.id) as transaction_count,
        COALESCE(SUM(CASE WHEN t.status = 'success' AND t.transaction_type = 'payment' AND t.amount < 0 THEN ABS(t.amount) ELSE 0 END), 0) as total_spent,
        MAX(t.timestamp) as last_transaction
    FROM users u
    LEFT JOIN wallets w ON u.id = w.user_id
    LEFT JOIN transactions t ON u.id = t.user_id
    {$where_clause}
    GROUP BY u.id
    ORDER BY u.created_at DESC
    LIMIT ? OFFSET ?
";

// Add pagination params
$params[] = $per_page;
$params[] = $offset;
$param_types .= 'ii';

$users_stmt = $conn->prepare($users_query);
if (!empty($param_types)) {
    $users_stmt->bind_param($param_types, ...$params);
}
$users_stmt->execute();
$users = $users_stmt->get_result()->fetch_all(MYSQLI_ASSOC);

// Get summary stats
$stats_query = "
    SELECT 
        COUNT(*) as total_users,
        SUM(CASE WHEN account_status = 'active' THEN 1 ELSE 0 END) as active_users,
        SUM(CASE WHEN account_status = 'suspended' THEN 1 ELSE 0 END) as inactive_users,
        SUM(CASE WHEN created_at >= CURDATE() THEN 1 ELSE 0 END) as new_today
    FROM users
";
$stats = $conn->query($stats_query)->fetch_assoc();

$conn->close();

$total_pages = ceil($total_users / $per_page);
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>User Management - FacePay Admin</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
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
        .stat-card.info { border-left-color: var(--info); }

        .stat-value {
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .stat-value.primary { color: var(--primary); }
        .stat-value.success { color: var(--success); }
        .stat-value.warning { color: var(--warning); }
        .stat-value.info { color: var(--info); }

        .stat-label {
            color: var(--text-light);
            font-size: 14px;
        }

        /* Search and Filters */
        .controls-section {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        }

        .search-filters {
            display: grid;
            grid-template-columns: 2fr 1fr auto;
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

        /* Users Table */
        .users-section {
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

        .users-table {
            width: 100%;
            border-collapse: collapse;
        }

        .users-table th {
            background: var(--bg-light);
            padding: 15px;
            text-align: left;
            font-weight: 600;
            color: var(--text-medium);
            border-bottom: 1px solid var(--border);
        }

        .users-table td {
            padding: 15px;
            border-bottom: 1px solid var(--border);
        }

        .users-table tbody tr.user-row:hover {
            background: #f0f4fa;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.08);
        }

        .user-info {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .user-avatar-table {
            width: 40px;
            height: 40px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 18px;
            color: #fff;
            margin-right: 10px;
            background: var(--primary);
        }

        .user-details h4 {
            margin: 0;
            font-size: 16px;
            font-weight: 600;
        }

        .user-id {
            font-size: 11px;
            color: #b0b7c3;
        }

        .user-email {
            color: #3182ce;
            text-decoration: none;
            font-weight: 500;
        }

        .user-joined, .wallet-spent {
            font-size: 12px;
            color: #b0b7c3;
        }

        .wallet-balance {
            font-weight: 700;
            color: #38a169;
            font-size: 15px;
        }

        .status-badge.active {
            background: #e6fffa;
            color: #38a169;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }

        .status-badge.inactive {
            background: #fff5f5;
            color: #e53e3e;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }

        .action-buttons {
            display: flex;
            gap: 8px;
        }

        .action-buttons .btn {
            border-radius: 50%;
            width: 36px;
            height: 36px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            transition: all 0.3s ease;
        }

        .action-buttons .btn:hover {
            opacity: 0.85;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.12);
        }

        .badge-joined {
            background: #f0f4fa;
            color: #667eea;
            border-radius: 8px;
            padding: 2px 8px;
            font-size: 11px;
            margin-left: 4px;
        }

        .transaction-label {
            color: #b0b7c3;
            font-size: 12px;
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

            .search-filters {
                grid-template-columns: 1fr;
            }

            .users-table {
                font-size: 14px;
            }

            .users-table th,
            .users-table td {
                padding: 10px;
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
                <a href="users.php" class="nav-item active">
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
                <h1 class="page-title">User Management</h1>
                <p class="page-subtitle">View customer accounts and manage access</p>
            </div>
        </div>

        <?php if ($message): ?>
            <div class="message <?= $message_type ?>">
                <i class="fas fa-<?= $message_type === 'success' ? 'check-circle' : 'exclamation-triangle' ?>"></i>
                <?= htmlspecialchars($message) ?>
            </div>
        <?php endif; ?>

        <!-- Stats -->
        <div class="stats-grid">
            <div class="stat-card primary">
                <div class="stat-value primary"><?= number_format($stats['total_users']) ?></div>
                <div class="stat-label">Total Users</div>
            </div>
            <div class="stat-card success">
                <div class="stat-value success"><?= number_format($stats['active_users']) ?></div>
                <div class="stat-label">Active Users</div>
            </div>
            <div class="stat-card warning">
                <div class="stat-value warning"><?= number_format($stats['inactive_users']) ?></div>
                <div class="stat-label">Blocked Users</div>
            </div>
            <div class="stat-card info">
                <div class="stat-value info"><?= number_format($stats['new_today']) ?></div>
                <div class="stat-label">New Today</div>
            </div>
        </div>

        <!-- Search and Filters -->
        <div class="controls-section">
            <form method="GET" class="search-filters">
                <div class="form-group">
                    <label class="form-label">Search Users</label>
                    <input type="text" class="form-input" name="search" 
                           value="<?= htmlspecialchars($search) ?>" 
                           placeholder="Search by name, email, or user ID...">
                </div>
                <div class="form-group">
                    <label class="form-label">Status Filter</label>
                    <select class="form-select" name="status">
                        <option value="">All Users</option>
                        <option value="active" <?= $status_filter === 'active' ? 'selected' : '' ?>>Active Only</option>
                        <option value="suspended" <?= $status_filter === 'suspended' ? 'selected' : '' ?>>Suspended Only</option>
                        <option value="pending" <?= $status_filter === 'pending' ? 'selected' : '' ?>>Pending Only</option>
                    </select>
                </div>
                <button type="submit" class="btn btn-primary">
                    <i class="fas fa-search"></i> Search
                </button>
            </form>
        </div>

        <!-- Users Table -->
        <div class="users-section">
            <div class="section-header">
                <div class="section-title">
                    <i class="fas fa-users"></i>
                    Users (<?= number_format($total_users) ?> total)
                </div>
            </div>

            <table class="users-table">
                <thead>
                    <tr>
                        <th>User</th>
                        <th>Contact</th>
                        <th>Wallet Balance</th>
                        <th>Transactions</th>
                        <th>Status</th>
                        <th>Last Activity</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    <?php if (empty($users)): ?>
                        <tr>
                            <td colspan="7" style="text-align: center; padding: 40px; color: var(--text-light);">
                                <i class="fas fa-users" style="font-size: 48px; margin-bottom: 15px; opacity: 0.3;"></i>
                                <br>No users found matching your criteria.
                            </td>
                        </tr>
                    <?php else: ?>
                        <?php foreach ($users as $user): ?>
                            <tr class="user-row">
                                <td>
                                    <div class="user-info">
                                        <div class="user-avatar-table">
                                            <?= strtoupper(substr($user['full_name'], 0, 2)) ?>
                                        </div>
                                        <div class="user-details">
                                            <h4><?= htmlspecialchars($user['full_name']) ?></h4>
                                            <span class="user-id">ID: <?= htmlspecialchars($user['user_id']) ?></span>
                                        </div>
                                    </div>
                                </td>
                                <td>
                                    <a href="mailto:<?= htmlspecialchars($user['email']) ?>" class="user-email">
                                        <?= htmlspecialchars($user['email']) ?>
                                    </a>
                                    <div class="user-joined">Joined <span class="badge-joined"><?= date('M j, Y', strtotime($user['created_at'])) ?></span></div>
                                </td>
                                <td>
                                    <span class="wallet-balance">RM <?= number_format($user['balance'] ?? 0, 2) ?></span>
                                    <div class="wallet-spent">Spent: <span>RM <?= number_format($user['total_spent'], 2) ?></span></div>
                                </td>
                                <td>
                                    <span class="transaction-count"><?= number_format($user['transaction_count']) ?></span> <span class="transaction-label">transactions</span>
                                </td>
                                <td>
                                    <span class="status-badge <?= $user['account_status'] === 'active' ? 'active' : 'inactive' ?>">
                                        <?= strtoupper($user['account_status']) ?>
                                    </span>
                                </td>
                                <td>
                                    <span class="last-activity">
                                        <?php if ($user['last_transaction']): ?>
                                            <?= date('M j, H:i', strtotime($user['last_transaction'])) ?>
                                        <?php else: ?>
                                            <span style="color: var(--text-light);">Never</span>
                                        <?php endif; ?>
                                    </span>
                                </td>
                                <td>
                                    <div class="action-buttons">
                                        <button class="btn btn-info" title="View Details" onclick="viewUser(<?= htmlspecialchars(json_encode($user)) ?>)">
                                            <i class="fas fa-eye"></i>
                                        </button>
                                        <?php if ($user['account_status'] === 'active'): ?>
                                            <button class="btn btn-error" title="Suspend User" onclick="toggleUserStatus(<?= $user['id'] ?>, '<?= htmlspecialchars($user['full_name']) ?>', 'suspend')">
                                                <i class="fas fa-ban"></i>
                                            </button>
                                        <?php else: ?>
                                            <button class="btn btn-success" title="Activate User" onclick="toggleUserStatus(<?= $user['id'] ?>, '<?= htmlspecialchars($user['full_name']) ?>', 'activate')">
                                                <i class="fas fa-check"></i>
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
                        <a href="?page=<?= $page - 1 ?>&search=<?= urlencode($search) ?>&status=<?= urlencode($status_filter) ?>">
                            <i class="fas fa-chevron-left"></i> Previous
                        </a>
                    <?php endif; ?>

                    <?php for ($i = max(1, $page - 2); $i <= min($total_pages, $page + 2); $i++): ?>
                        <?php if ($i == $page): ?>
                            <span class="current"><?= $i ?></span>
                        <?php else: ?>
                            <a href="?page=<?= $i ?>&search=<?= urlencode($search) ?>&status=<?= urlencode($status_filter) ?>"><?= $i ?></a>
                        <?php endif; ?>
                    <?php endfor; ?>

                    <?php if ($page < $total_pages): ?>
                        <a href="?page=<?= $page + 1 ?>&search=<?= urlencode($search) ?>&status=<?= urlencode($status_filter) ?>">
                            Next <i class="fas fa-chevron-right"></i>
                        </a>
                    <?php endif; ?>
                </div>
            <?php endif; ?>
        </div>
    </main>

    <!-- User Details Modal -->
    <div id="userModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 class="modal-title">User Details</h3>
                <button class="modal-close" onclick="closeModal('userModal')">&times;</button>
            </div>
            <div class="modal-body" id="userModalBody">
                <!-- User details will be loaded here -->
            </div>
            <div class="modal-footer">
                <button type="button" class="btn-cancel" onclick="closeModal('userModal')">Close</button>
            </div>
        </div>
    </div>

    <!-- User Status Toggle Modal -->
    <div id="statusModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 class="modal-title">Change User Status</h3>
                <button class="modal-close" onclick="closeModal('statusModal')">&times;</button>
            </div>
            <form method="POST" id="statusForm">
                <input type="hidden" name="user_id" id="status_user_id">
                <div class="modal-body">
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <strong>User:</strong> <span id="status_user_name"></span>
                    </div>

                    <div id="status_message" style="font-size: 16px; margin-bottom: 20px;">
                        <!-- Status change message will be inserted here -->
                    </div>

                    <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 8px; font-size: 14px;">
                        <strong>⚠️ Impact:</strong> <span id="status_impact"></span>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn-cancel" onclick="closeModal('statusModal')">Cancel</button>
                    <button type="submit" class="btn-save" id="status_submit_btn">Confirm</button>
                </div>
            </form>
        </div>
    </div>

    <script>
        // Modal functions
        function closeModal(modalId) {
            document.getElementById(modalId).classList.remove('show');
        }

        function viewUser(user) {
            const modalBody = document.getElementById('userModalBody');
            
            const lastTransaction = user.last_transaction ? 
                new Date(user.last_transaction).toLocaleString() : 'Never';
            
            modalBody.innerHTML = `
                <div style="display: grid; gap: 20px;">
                    <div>
                        <h4 style="margin-bottom: 15px; color: var(--primary);">Personal Information</h4>
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; line-height: 1.8;">
                            <strong>Full Name:</strong> ${user.full_name}<br>
                            <strong>Email:</strong> ${user.email}<br>
                            <strong>User ID:</strong> ${user.user_id}<br>
                            <strong>Date Registered:</strong> ${new Date(user.created_at).toLocaleDateString()}
                        </div>
                    </div>

                    <div>
                        <h4 style="margin-bottom: 15px; color: var(--success);">Account Status</h4>
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; line-height: 1.8;">
                            <strong>Status:</strong> <span class="status-badge ${user.account_status === 'active' ? 'active' : 'inactive'}">${user.account_status}</span><br>
                            <strong>PIN Status:</strong> ${user.pin_code ? 'Set' : 'Not Set'}<br>
                            <strong>Email Verified:</strong> ${user.is_verified ? 'Yes' : 'No'}
                        </div>
                    </div>

                    <div>
                        <h4 style="margin-bottom: 15px; color: var(--warning);">Wallet & Activity</h4>
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; line-height: 1.8;">
                            <strong>Current Balance:</strong> RM ${parseFloat(user.balance || 0).toFixed(2)}<br>
                            <strong>Total Spent:</strong> RM ${parseFloat(user.total_spent || 0).toFixed(2)}<br>
                            <strong>Total Transactions:</strong> ${user.transaction_count}<br>
                            <strong>Last Transaction:</strong> ${lastTransaction}
                        </div>
                    </div>
                </div>
            `;
            
            document.getElementById('userModal').classList.add('show');
        }

        function toggleUserStatus(userId, userName, action) {
            const isActivating = action === 'activate';
            
            document.getElementById('status_user_id').value = userId;
            document.getElementById('status_user_name').textContent = userName;
            
            // Set the correct form action
            document.getElementById('statusForm').querySelector('input[name="action"]')?.remove();
            const actionInput = document.createElement('input');
            actionInput.type = 'hidden';
            actionInput.name = 'action';
            actionInput.value = isActivating ? 'activate_user' : 'suspend_user';
            document.getElementById('statusForm').appendChild(actionInput);
            
            const statusMessage = document.getElementById('status_message');
            const statusImpact = document.getElementById('status_impact');
            const submitBtn = document.getElementById('status_submit_btn');
            
            if (isActivating) {
                statusMessage.innerHTML = `<p>Are you sure you want to <strong style="color: var(--success);">activate</strong> this user?</p>`;
                statusImpact.textContent = 'This user will be able to login and make payments again.';
                submitBtn.textContent = 'Activate User';
                submitBtn.style.background = 'var(--success)';
            } else {
                statusMessage.innerHTML = `<p>Are you sure you want to <strong style="color: var(--error);">suspend</strong> this user?</p>`;
                statusImpact.textContent = 'This user will be unable to login or make payments until reactivated.';
                submitBtn.textContent = 'Suspend User';
                submitBtn.style.background = 'var(--error)';
            }
            
            document.getElementById('statusModal').classList.add('show');
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

        // Search auto-submit on Enter
        document.querySelector('input[name="search"]').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                this.closest('form').submit();
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            // Escape to close any open modal
            if (e.key === 'Escape') {
                document.querySelectorAll('.modal.show').forEach(modal => {
                    modal.classList.remove('show');
                });
            }
            
            // Ctrl+F to focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
                e.preventDefault();
                document.querySelector('input[name="search"]').focus();
            }
        });

        // Loading state for status form
        document.getElementById('statusForm').addEventListener('submit', function() {
            const submitBtn = document.getElementById('status_submit_btn');
            const originalText = submitBtn.textContent;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            submitBtn.disabled = true;
            
            // Re-enable after 10 seconds in case of errors
            setTimeout(() => {
                submitBtn.textContent = originalText;
                submitBtn.disabled = false;
            }, 10000);
        });

        console.log('User Management System loaded successfully');
        console.log('Available shortcuts: Escape (Close Modal), Ctrl+F (Focus Search)');
    </script>
</body>
</html>