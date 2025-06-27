<?php
// admin/merchants.php
require_once 'config/admin_config.php';

// Require authentication
$admin = requireAdminAuth(['super_admin', 'kiosk_admin']);

$message = '';
$message_type = '';

// Handle form submissions
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $action = $_POST['action'] ?? '';
    
    try {
        $conn = dbConnect();
        $auth = getAdminAuth();
        
        if ($action === 'add_merchant') {
            $merchant_id = strtoupper(trim($_POST['merchant_id']));
            $name = trim($_POST['name']);
            $return_url = trim($_POST['return_url'] ?? '');
            $cancel_url = trim($_POST['cancel_url'] ?? '');
            $webhook_url = trim($_POST['webhook_url'] ?? '');
            
            // Generate API key
            $api_key = 'sk_' . bin2hex(random_bytes(24));
            
            // Check if merchant_id already exists
            $check_stmt = $conn->prepare("SELECT id FROM merchants WHERE merchant_id = ?");
            $check_stmt->bind_param("s", $merchant_id);
            $check_stmt->execute();
            if ($check_stmt->get_result()->num_rows > 0) {
                throw new Exception("Merchant ID already exists");
            }
            
            // Insert new merchant with URLs
            $stmt = $conn->prepare("
                INSERT INTO merchants (merchant_id, name, api_key, return_url, cancel_url, webhook_url, is_active)
                VALUES (?, ?, ?, ?, ?, ?, 1)
            ");
            $stmt->bind_param("ssssss", $merchant_id, $name, $api_key, $return_url, $cancel_url, $webhook_url);
            $stmt->execute();
            
            // Log action
            $auth->logAuditAction($admin['id'], 'create_merchant', 'merchant', $merchant_id, [
                'name' => $name
            ]);
            
            $message = "Merchant '{$name}' added successfully!";
            $message_type = 'success';
            
        } elseif ($action === 'update_merchant') {
            $id = intval($_POST['merchant_db_id']);
            $name = trim($_POST['name']);
            $return_url = trim($_POST['return_url'] ?? '');
            $cancel_url = trim($_POST['cancel_url'] ?? '');
            $webhook_url = trim($_POST['webhook_url'] ?? '');
            $is_active = isset($_POST['is_active']) ? 1 : 0;
            
            $stmt = $conn->prepare("
                UPDATE merchants 
                SET name = ?, return_url = ?, cancel_url = ?, webhook_url = ?, is_active = ?
                WHERE id = ?
            ");
            $stmt->bind_param("sssdii", $name, $return_url, $cancel_url, $webhook_url, $is_active, $id);
            $stmt->execute();
            
            // Log action
            $auth->logAuditAction($admin['id'], 'update_merchant', 'merchant', $id, [
                'name' => $name,
                'is_active' => $is_active
            ]);
            
            $message = "Merchant updated successfully!";
            $message_type = 'success';
            
        } elseif ($action === 'regenerate_api_key') {
            $id = intval($_POST['merchant_id']);
            $new_api_key = 'sk_' . bin2hex(random_bytes(24));
            
            $stmt = $conn->prepare("UPDATE merchants SET api_key = ? WHERE id = ?");
            $stmt->bind_param("si", $new_api_key, $id);
            $stmt->execute();
            
            // Log action
            $auth->logAuditAction($admin['id'], 'regenerate_api_key', 'merchant', $id, []);
            
            $message = "API key regenerated successfully!";
            $message_type = 'success';
        }
        
        $conn->close();
        
    } catch (Exception $e) {
        $message = "Error: " . $e->getMessage();
        $message_type = 'error';
    }
}

// Get all merchants with statistics (FIXED for payment gateway)
$conn = dbConnect();
$merchants_query = "
    SELECT 
        m.*,
        COUNT(DISTINCT t.id) as total_transactions,
        COALESCE(SUM(CASE WHEN DATE(t.timestamp) = CURRENT_DATE() AND t.status = 'success' THEN ABS(t.amount) ELSE 0 END), 0) as today_revenue,
        COALESCE(SUM(CASE WHEN t.status = 'success' THEN ABS(t.amount) ELSE 0 END), 0) as total_revenue
    FROM merchants m
    LEFT JOIN transactions t ON m.merchant_id = t.merchant_id  -- Direct join
    GROUP BY m.id
    ORDER BY m.id DESC
";
$merchants = $conn->query($merchants_query)->fetch_all(MYSQLI_ASSOC);
$conn->close();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Merchant Management - FacePay Admin</title>
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

        .add-btn {
            background: linear-gradient(135deg, var(--success), #48bb78);
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

        .add-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(56, 161, 105, 0.3);
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
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            text-align: center;
        }

        .stat-value {
            font-size: 24px;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 5px;
        }

        .stat-label {
            color: var(--text-light);
            font-size: 14px;
        }

        /* Merchants Table */
        .merchants-section {
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

        .merchants-table {
            width: 100%;
            border-collapse: collapse;
        }

        .merchants-table th {
            background: var(--bg-light);
            padding: 15px;
            text-align: left;
            font-weight: 600;
            color: var(--text-medium);
            border-bottom: 1px solid var(--border);
        }

        .merchants-table td {
            padding: 15px;
            border-bottom: 1px solid var(--border);
        }

        .merchants-table tbody tr:hover {
            background: var(--bg-light);
        }

        .merchant-info {
            display: flex;
            align-items: center;
            gap: 12px;
        }

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
        }

        .merchant-details h4 {
            font-weight: 600;
            color: var(--text-dark);
            margin-bottom: 2px;
        }

        .merchant-details span {
            font-size: 12px;
            color: var(--text-light);
        }

        .status-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }

        .status-badge.active {
            background: rgba(56, 161, 105, 0.1);
            color: var(--success);
        }

        .status-badge.inactive {
            background: rgba(229, 62, 62, 0.1);
            color: var(--error);
        }

        .action-buttons {
            display: flex;
            gap: 8px;
        }

        .btn {
            padding: 8px 12px;
            border: none;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }

        .btn-primary {
            background: var(--primary);
            color: white;
        }

        .btn-warning {
            background: var(--warning);
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
            animation: fadeIn 0.3s ease;
        }

        .modal.show {
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .modal-content {
            background: white;
            border-radius: 16px;
            max-width: 700px;
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

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-group.full-width {
            grid-column: span 2;
        }

        .form-label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: var(--text-dark);
        }

        .form-input {
            width: 100%;
            padding: 12px;
            border: 2px solid var(--border);
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s ease;
        }

        .form-input:focus {
            outline: none;
            border-color: var(--primary);
        }

        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 8px;
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

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
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

            .page-header {
                flex-direction: column;
                gap: 15px;
                align-items: flex-start;
            }

            .form-grid {
                grid-template-columns: 1fr;
            }

            .form-group.full-width {
                grid-column: span 1;
            }

            .merchants-table {
                font-size: 14px;
            }

            .merchants-table th,
            .merchants-table td {
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
                <a href="merchants.php" class="nav-item active">
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
                <h1 class="page-title">Merchant Management</h1>
                <p class="page-subtitle">Manage payment gateway partners and their integration settings</p>
            </div>
            <button class="add-btn" onclick="openAddMerchantModal()">
                <i class="fas fa-plus"></i> Add Merchant
            </button>
        </div>

        <?php if ($message): ?>
            <div class="message <?= $message_type ?>">
                <i class="fas fa-<?= $message_type === 'success' ? 'check-circle' : 'exclamation-triangle' ?>"></i>
                <?= htmlspecialchars($message) ?>
            </div>
        <?php endif; ?>

        <!-- Stats -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value"><?= count($merchants) ?></div>
                <div class="stat-label">Total Merchants</div>
            </div>
            <div class="stat-card">
                <div class="stat-value"><?= count(array_filter($merchants, fn($m) => $m['is_active'])) ?></div>
                <div class="stat-label">Active Merchants</div>
            </div>
            <div class="stat-card">
                <div class="stat-value"><?= array_sum(array_column($merchants, 'total_transactions')) ?></div>
                <div class="stat-label">Total Payments</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">RM <?= number_format(array_sum(array_column($merchants, 'today_revenue')), 2) ?></div>
                <div class="stat-label">Today's Revenue</div>
            </div>
        </div>

        <!-- Merchants Table -->
        <div class="merchants-section">
            <div class="section-header">
                <div class="section-title">
                    <i class="fas fa-store"></i>
                    Registered Merchants
                </div>
            </div>

            <table class="merchants-table">
                <thead>
                    <tr>
                        <th>Merchant</th>
                        <th>Integration URLs</th>
                        <th>Payments</th>
                        <th>Revenue</th>
                        <th>Status</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    <?php if (empty($merchants)): ?>
                        <tr>
                            <td colspan="6" style="text-align: center; padding: 40px; color: var(--text-light);">
                                <i class="fas fa-store" style="font-size: 48px; margin-bottom: 15px; opacity: 0.3;"></i>
                                <br>No merchants registered yet. Add your first merchant to get started!
                            </td>
                        </tr>
                    <?php else: ?>
                        <?php foreach ($merchants as $merchant): ?>
                            <tr>
                                <td>
                                    <div class="merchant-info">
                                        <div class="merchant-avatar">
                                            <?= strtoupper(substr($merchant['name'], 0, 2)) ?>
                                        </div>
                                        <div class="merchant-details">
                                            <h4><?= htmlspecialchars($merchant['name']) ?></h4>
                                            <span><?= htmlspecialchars($merchant['merchant_id']) ?></span>
                                        </div>
                                    </div>
                                </td>
                                <td>
                                    <div style="font-size: 13px;">
                                        <div><strong>Return:</strong> <?= htmlspecialchars(substr($merchant['return_url'] ?? 'Not set', 0, 30)) ?><?= strlen($merchant['return_url'] ?? '') > 30 ? '...' : '' ?></div>
                                        <div><strong>Cancel:</strong> <?= htmlspecialchars(substr($merchant['cancel_url'] ?? 'Not set', 0, 30)) ?><?= strlen($merchant['cancel_url'] ?? '') > 30 ? '...' : '' ?></div>
                                    </div>
                                </td>
                                <td>
                                    <strong><?= $merchant['total_transactions'] ?></strong> payments
                                </td>
                                <td>
                                    <div>
                                        <strong>RM <?= number_format($merchant['total_revenue'], 2) ?></strong><br>
                                        <small style="color: var(--text-light);">Total</small>
                                    </div>
                                </td>
                                <td>
                                    <span class="status-badge <?= $merchant['is_active'] ? 'active' : 'inactive' ?>">
                                        <?= $merchant['is_active'] ? 'Active' : 'Inactive' ?>
                                    </span>
                                </td>
                                <td>
                                    <div class="action-buttons">
                                        <button class="btn btn-primary" onclick="editMerchant(<?= htmlspecialchars(json_encode($merchant)) ?>)">
                                            <i class="fas fa-edit"></i> Edit
                                        </button>
                                        <button class="btn btn-info" onclick="viewApiKey('<?= htmlspecialchars($merchant['api_key']) ?>')">
                                            <i class="fas fa-key"></i> API
                                        </button>
                                        <button class="btn btn-warning" onclick="regenerateApiKey(<?= $merchant['id'] ?>)">
                                            <i class="fas fa-sync"></i>
                                        </button>
                                    </div>
                                </td>
                            </tr>
                        <?php endforeach; ?>
                    <?php endif; ?>
                </tbody>
            </table>
        </div>
    </main>

    <!-- Add Merchant Modal -->
    <div id="addMerchantModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 class="modal-title">Add New Merchant</h3>
                <button class="modal-close" onclick="closeModal('addMerchantModal')">&times;</button>
            </div>
            <form method="POST" id="addMerchantForm">
                <input type="hidden" name="action" value="add_merchant">
                <div class="modal-body">
                    <div class="form-grid">
                        <div class="form-group">
                            <label class="form-label" for="merchant_id">Merchant ID</label>
                            <input type="text" class="form-input" id="merchant_id" name="merchant_id" required
                                   placeholder="e.g., MCD, KFC, BK">
                        </div>
                        <div class="form-group">
                            <label class="form-label" for="name">Business Name</label>
                            <input type="text" class="form-input" id="name" name="name" required
                                   placeholder="e.g., McDonald's Malaysia">
                        </div>
                        <div class="form-group full-width">
                            <label class="form-label" for="return_url">Return URL (Success)</label>
                            <input type="url" class="form-input" id="return_url" name="return_url"
                                   placeholder="https://mcdonalds.com/payment/success">
                        </div>
                        <div class="form-group full-width">
                            <label class="form-label" for="cancel_url">Cancel URL (Failed/Cancelled)</label>
                            <input type="url" class="form-input" id="cancel_url" name="cancel_url"
                                   placeholder="https://mcdonalds.com/payment/cancel">
                        </div>
                        <!-- <div class="form-group full-width">
                            <label class="form-label" for="webhook_url">Webhook URL (Optional)</label>
                            <input type="url" class="form-input" id="webhook_url" name="webhook_url"
                                   placeholder="https://mcdonalds.com/webhooks/facepay">
                        </div> -->
                        <div class="form-group full-width">
                            <div class="checkbox-group">
                                <input type="checkbox" id="is_active" name="is_active" value="1" checked>
                                <label for="is_active">Active (can process payments)</label>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn-cancel" onclick="closeModal('addMerchantModal')">Cancel</button>
                    <button type="submit" class="btn-save">Add Merchant</button>
                </div>
            </form>
        </div>
    </div>

    <!-- Edit Merchant Modal -->
    <div id="editMerchantModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 class="modal-title">Edit Merchant</h3>
                <button class="modal-close" onclick="closeModal('editMerchantModal')">&times;</button>
            </div>
            <form method="POST" id="editMerchantForm">
                <input type="hidden" name="action" value="update_merchant">
                <input type="hidden" name="merchant_db_id" id="edit_merchant_db_id">
                <div class="modal-body">
                    <div class="form-grid">
                        <div class="form-group full-width">
                            <label class="form-label" for="edit_name">Business Name</label>
                            <input type="text" class="form-input" id="edit_name" name="name" required>
                        </div>
                        <div class="form-group full-width">
                            <label class="form-label" for="edit_return_url">Return URL (Success)</label>
                            <input type="url" class="form-input" id="edit_return_url" name="return_url">
                        </div>
                        <div class="form-group full-width">
                            <label class="form-label" for="edit_cancel_url">Cancel URL (Failed/Cancelled)</label>
                            <input type="url" class="form-input" id="edit_cancel_url" name="cancel_url">
                        </div>
                        <!-- <div class="form-group full-width">
                            <label class="form-label" for="edit_webhook_url">Webhook URL (Optional)</label>
                            <input type="url" class="form-input" id="edit_webhook_url" name="webhook_url">
                        </div> -->
                        <div class="form-group full-width">
                            <div class="checkbox-group">
                                <input type="checkbox" id="edit_is_active" name="is_active" value="1">
                                <label for="edit_is_active">Active (can process payments)</label>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn-cancel" onclick="closeModal('editMerchantModal')">Cancel</button>
                    <button type="submit" class="btn-save">Update Merchant</button>
                </div>
            </form>
        </div>
    </div>

    <!-- API Key Modal -->
    <div id="apiKeyModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 class="modal-title">API Integration Details</h3>
                <button class="modal-close" onclick="closeModal('apiKeyModal')">&times;</button>
            </div>
            <div class="modal-body">
                <div class="form-group">
                    <label class="form-label">API Key (Keep this secure!)</label>
                    <div style="display: flex; gap: 10px; align-items: center;">
                        <input type="text" class="form-input" id="apiKeyDisplay" readonly 
                               style="font-family: monospace; background: #f8f9fa;">
                        <button type="button" class="btn btn-info" onclick="copyApiKey()">
                            <i class="fas fa-copy"></i> Copy
                        </button>
                    </div>
                    <small style="color: var(--text-light); margin-top: 8px; display: block;">
                        Use this API key in your payment integration to authenticate with FacePay.
                    </small>
                </div>
                <div class="form-group">
                    <label class="form-label">Payment Gateway Integration</label>
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; font-size: 13px; line-height: 1.5;">
                        <strong>🚀 Payment Gateway Integration (Like PayPal):</strong><br><br>
                        
                        <strong>1. Payment Button on Your Site:</strong><br>
                        <code>&lt;form action="https://facepay.com/gateway/checkout.php" method="POST"&gt;<br>
                        &nbsp;&nbsp;&lt;input type="hidden" name="merchant_id" value="[YOUR_MERCHANT_ID]"&gt;<br>
                        &nbsp;&nbsp;&lt;input type="hidden" name="api_key" value="[YOUR_API_KEY]"&gt;<br>
                        &nbsp;&nbsp;&lt;input type="hidden" name="return_url" value="[YOUR_SUCCESS_URL]"&gt;<br>
                        &nbsp;&nbsp;&lt;input type="hidden" name="cancel_url" value="[YOUR_CANCEL_URL]"&gt;<br>
                        &nbsp;&nbsp;&lt;button type="submit"&gt;Pay with FacePay&lt;/button&gt;<br>
                        &lt;/form&gt;</code><br><br>
                        
                        <strong>2. Customer Flow:</strong><br>
                        • Customer clicks "Pay with FacePay"<br>
                        • Redirected to FacePay for face scan + PIN<br>
                        • After payment, returned to your success/cancel URL<br><br>
                        
                        <strong>3. Return Parameters:</strong><br>
                        Success: <code>your-site.com/success?payment_id=PAY123&status=success&amount=25.50</code><br>
                        Failed: <code>your-site.com/cancel?error=insufficient_funds&order_id=ORDER123</code>
                    </div>
                </div>
                <!-- <div class="form-group">
                    <label class="form-label">Advanced API Endpoints (Optional)</label>
                    <div style="background: #e8f4f8; padding: 15px; border-radius: 8px; font-size: 13px; line-height: 1.5;">
                        <strong>🔧 Direct API Integration:</strong><br><br>
                        
                        <strong>Face Recognition:</strong><br>
                        POST https://facepay.com/api/authenticate<br>
                        Headers: Authorization: Bearer [API_KEY]<br><br>
                        
                        <strong>Payment Processing:</strong><br>
                        POST https://facepay.com/api/process_payment<br>
                        Headers: Authorization: Bearer [API_KEY]<br><br>
                        
                        <strong>Transaction Status:</strong><br>
                        GET https://facepay.com/api/transaction_status?id=[PAYMENT_ID]<br>
                        Headers: Authorization: Bearer [API_KEY]
                    </div>
                </div>
            </div> -->
            <div class="modal-footer">
                <button type="button" class="btn-cancel" onclick="closeModal('apiKeyModal')">Close</button>
            </div>
        </div>
    </div>

    <script>
        // Modal functions
        function openAddMerchantModal() {
            document.getElementById('addMerchantModal').classList.add('show');
        }

        function closeModal(modalId) {
            document.getElementById(modalId).classList.remove('show');
        }

        function editMerchant(merchant) {
            document.getElementById('edit_merchant_db_id').value = merchant.id;
            document.getElementById('edit_name').value = merchant.name;
            document.getElementById('edit_return_url').value = merchant.return_url || '';
            document.getElementById('edit_cancel_url').value = merchant.cancel_url || '';
            // Only set webhook_url if the input exists (uncomment if you add the input in the modal)
            // var webhookInput = document.getElementById('edit_webhook_url');
            // if (webhookInput) webhookInput.value = merchant.webhook_url || '';
            document.getElementById('edit_is_active').checked = merchant.is_active == 1;
            
            document.getElementById('editMerchantModal').classList.add('show');
        }

        function viewApiKey(apiKey) {
            document.getElementById('apiKeyDisplay').value = apiKey;
            document.getElementById('apiKeyModal').classList.add('show');
        }

        function copyApiKey() {
            const apiKeyInput = document.getElementById('apiKeyDisplay');
            apiKeyInput.select();
            apiKeyInput.setSelectionRange(0, 99999); // For mobile devices
            
            try {
                document.execCommand('copy');
                
                // Show feedback
                const button = event.target.closest('button');
                const originalText = button.innerHTML;
                button.innerHTML = '<i class="fas fa-check"></i> Copied!';
                button.style.background = 'var(--success)';
                
                setTimeout(() => {
                    button.innerHTML = originalText;
                    button.style.background = 'var(--info)';
                }, 2000);
            } catch (err) {
                console.error('Failed to copy: ', err);
                alert('Failed to copy API key. Please select and copy manually.');
            }
        }

        function regenerateApiKey(merchantId) {
            if (confirm('Are you sure you want to regenerate the API key?\n\nThis will immediately invalidate the current API key and may break existing integrations until updated.')) {
                const form = document.createElement('form');
                form.method = 'POST';
                form.innerHTML = `
                    <input type="hidden" name="action" value="regenerate_api_key">
                    <input type="hidden" name="merchant_id" value="${merchantId}">
                `;
                document.body.appendChild(form);
                form.submit();
            }
        }

        // Auto-uppercase merchant ID and validate format
        document.getElementById('merchant_id').addEventListener('input', function(e) {
            e.target.value = e.target.value.toUpperCase().replace(/[^A-Z0-9\-_]/g, '');
        });

        // Close modals when clicking outside
        document.addEventListener('click', function(e) {
            if (e.target.classList.contains('modal')) {
                e.target.classList.remove('show');
            }
        });

        // Enhanced form validation
        document.getElementById('addMerchantForm').addEventListener('submit', function(e) {
            const merchantId = document.getElementById('merchant_id').value;
            const name = document.getElementById('name').value;
            
            if (merchantId.length < 2) {
                e.preventDefault();
                alert('Merchant ID must be at least 2 characters long');
                return;
            }
            
            if (name.length < 3) {
                e.preventDefault();
                alert('Business name must be at least 3 characters long');
                return;
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

        // Add loading state to form submissions
        document.getElementById('addMerchantForm').addEventListener('submit', function() {
            const submitBtn = this.querySelector('.btn-save');
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Adding...';
            submitBtn.disabled = true;
        });

        document.getElementById('editMerchantForm').addEventListener('submit', function() {
            const submitBtn = this.querySelector('.btn-save');
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Updating...';
            submitBtn.disabled = true;
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            // Ctrl+M or Cmd+M to add new merchant
            if ((e.ctrlKey || e.metaKey) && e.key === 'm') {
                e.preventDefault();
                openAddMerchantModal();
            }
            
            // Escape to close any open modal
            if (e.key === 'Escape') {
                document.querySelectorAll('.modal.show').forEach(modal => {
                    modal.classList.remove('show');
                });
            }
        });

        // Initialize tooltips or help text
        document.addEventListener('DOMContentLoaded', function() {
            // Add helpful tooltips
            const merchantIdInput = document.getElementById('merchant_id');
            if (merchantIdInput) {
                merchantIdInput.title = 'Use a short, unique identifier like MCD for McDonald\'s or KFC for KFC';
            }
        });

        console.log('FacePay Merchant Management System loaded successfully');
        console.log('Available keyboard shortcuts: Ctrl+M (Add Merchant), Escape (Close Modal)');
    </script>
</body>
</html>