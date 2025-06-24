<?php
// export_merchant_report.php (SIMPLIFIED VERSION)
require_once 'config/admin_config.php';

// Require authentication
$admin = requireAdminAuth(['super_admin', 'admin', 'finance']);

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    die('Invalid request method');
}

// Get form parameters
$merchant_id = $_POST['export_merchant_id'] ?? '';
$date_from = $_POST['export_date_from'] ?? date('Y-m-d', strtotime('-30 days'));
$date_to = $_POST['export_date_to'] ?? date('Y-m-d');

if (empty($merchant_id)) {
    die('Please select a merchant');
}

try {
    $conn = dbConnect();
    
    // Get merchant details
    $merchant_stmt = $conn->prepare("
        SELECT merchant_id, merchant_name 
        FROM transactions 
        WHERE merchant_id = ? 
        LIMIT 1
    ");
    $merchant_stmt->bind_param("s", $merchant_id);
    $merchant_stmt->execute();
    $merchant_result = $merchant_stmt->get_result();
    $merchant = $merchant_result->fetch_assoc();
    $merchant_stmt->close();
    
    if (!$merchant) {
        die('Merchant not found');
    }
    
    // Get transaction data for the merchant
    $transactions_stmt = $conn->prepare("
        SELECT 
            t.*,
            u.full_name,
            u.email
        FROM transactions t
        LEFT JOIN users u ON t.user_id = u.id
        WHERE t.merchant_id = ? 
        AND DATE(t.timestamp) BETWEEN ? AND ?
        ORDER BY t.timestamp DESC
    ");
    $transactions_stmt->bind_param("sss", $merchant_id, $date_from, $date_to);
    $transactions_stmt->execute();
    $transactions = $transactions_stmt->get_result()->fetch_all(MYSQLI_ASSOC);
    $transactions_stmt->close();
    
    // Calculate SIMPLIFIED summary statistics (NO PLATFORM FEES)
    $total_transactions = count($transactions);
    $successful_transactions = array_filter($transactions, fn($t) => $t['status'] === 'success' && $t['transaction_type'] === 'payment');
    $failed_transactions = array_filter($transactions, fn($t) => $t['status'] === 'failed');
    $refunded_transactions = array_filter($transactions, fn($t) => $t['status'] === 'refunded');
    
    $total_revenue = array_sum(array_map(fn($t) => $t['status'] === 'success' && $t['transaction_type'] === 'payment' ? abs($t['amount']) : 0, $transactions));
    $total_refunds = array_sum(array_map(fn($t) => $t['status'] === 'refunded' || $t['transaction_type'] === 'refund' ? abs($t['amount']) : 0, $transactions));
    $net_revenue = $total_revenue - $total_refunds;
    
    $success_rate = $total_transactions > 0 ? (count($successful_transactions) / $total_transactions) * 100 : 0;
    $avg_transaction = count($successful_transactions) > 0 ? $total_revenue / count($successful_transactions) : 0;
    
    // Get daily breakdown
    $daily_stmt = $conn->prepare("
        SELECT 
            DATE(timestamp) as date,
            COUNT(*) as transaction_count,
            SUM(CASE WHEN status = 'success' AND transaction_type = 'payment' THEN ABS(amount) ELSE 0 END) as revenue,
            SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_count,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_count
        FROM transactions
        WHERE merchant_id = ? 
        AND DATE(timestamp) BETWEEN ? AND ?
        GROUP BY DATE(timestamp)
        ORDER BY date ASC
    ");
    $daily_stmt->bind_param("sss", $merchant_id, $date_from, $date_to);
    $daily_stmt->execute();
    $daily_data = $daily_stmt->get_result()->fetch_all(MYSQLI_ASSOC);
    $daily_stmt->close();
    
    $conn->close();
    
    // Generate simplified PDF report
    generateSimplifiedPDF($merchant, $transactions, $daily_data, $total_revenue, $total_refunds, $net_revenue, $success_rate, $avg_transaction, $date_from, $date_to);
    
} catch (Exception $e) {
    error_log("Export error: " . $e->getMessage());
    die('Error generating report: ' . $e->getMessage());
}

function generateSimplifiedPDF($merchant, $transactions, $daily_data, $total_revenue, $total_refunds, $net_revenue, $success_rate, $avg_transaction, $date_from, $date_to) {
    $filename = "merchant_report_{$merchant['merchant_id']}_{$date_from}_to_{$date_to}.html";
    
    header('Content-Type: text/html');
    header('Content-Disposition: inline; filename="' . $filename . '"');
    
    ?>
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Merchant Financial Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; color: #333; }
            .header { text-align: center; border-bottom: 3px solid #667eea; padding-bottom: 20px; margin-bottom: 30px; }
            .merchant-name { font-size: 28px; color: #667eea; margin-bottom: 10px; }
            .period { font-size: 16px; color: #666; }
            .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
            .summary-card { background: #f8f9fa; border-left: 4px solid #667eea; padding: 20px; border-radius: 5px; }
            .card-title { font-size: 14px; color: #666; margin-bottom: 5px; }
            .card-value { font-size: 24px; font-weight: bold; color: #333; }
            .revenue { border-left-color: #38a169; }
            .success { border-left-color: #38a169; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background: #667eea; color: white; }
            tr:nth-child(even) { background: #f9f9f9; }
            .section-title { font-size: 20px; color: #667eea; margin: 30px 0 15px 0; border-bottom: 2px solid #667eea; padding-bottom: 5px; }
            .positive { color: #27ae60; font-weight: bold; }
            .negative { color: #e74c3c; font-weight: bold; }
            @media print { body { margin: 0; } }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="merchant-name"><?= htmlspecialchars($merchant['merchant_name']) ?></div>
            <div>Merchant ID: <?= htmlspecialchars($merchant['merchant_id']) ?></div>
            <div class="period">Financial Report: <?= $date_from ?> to <?= $date_to ?></div>
            <div style="font-size: 12px; color: #999; margin-top: 10px;">Generated on <?= date('F j, Y \a\t g:i A') ?> by FacePay Admin</div>
        </div>

        <!-- SIMPLIFIED Financial Summary -->
        <div class="section-title">📊 Financial Summary</div>
        <div class="summary-grid">
            <div class="summary-card revenue">
                <div class="card-title">Total Revenue</div>
                <div class="card-value">RM <?= number_format($total_revenue, 2) ?></div>
            </div>
            <?php if ($total_refunds > 0): ?>
            <div class="summary-card">
                <div class="card-title">Total Refunds</div>
                <div class="card-value negative">RM <?= number_format($total_refunds, 2) ?></div>
            </div>
            <div class="summary-card revenue">
                <div class="card-title">Net Revenue</div>
                <div class="card-value positive">RM <?= number_format($net_revenue, 2) ?></div>
            </div>
            <?php endif; ?>
            <div class="summary-card success">
                <div class="card-title">Success Rate</div>
                <div class="card-value"><?= number_format($success_rate, 1) ?>%</div>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px;">
            <div class="summary-card">
                <div class="card-title">Total Transactions</div>
                <div class="card-value"><?= count($transactions) ?></div>
            </div>
            <div class="summary-card">
                <div class="card-title">Average Transaction</div>
                <div class="card-value">RM <?= number_format($avg_transaction, 2) ?></div>
            </div>
        </div>

        <?php if (!empty($daily_data)): ?>
        <!-- Daily Performance -->
        <div class="section-title">📈 Daily Performance</div>
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Revenue</th>
                    <th>Transactions</th>
                    <th>Success Rate</th>
                </tr>
            </thead>
            <tbody>
                <?php foreach ($daily_data as $day): 
                    $day_success_rate = $day['transaction_count'] > 0 ? ($day['successful_count'] / $day['transaction_count']) * 100 : 0;
                ?>
                <tr>
                    <td><?= date('M j, Y', strtotime($day['date'])) ?></td>
                    <td class="positive">RM <?= number_format($day['revenue'], 2) ?></td>
                    <td><?= $day['transaction_count'] ?></td>
                    <td><?= number_format($day_success_rate, 1) ?>%</td>
                </tr>
                <?php endforeach; ?>
            </tbody>
        </table>
        <?php endif; ?>

        <!-- SIMPLIFIED Transaction Details -->
        <div class="section-title">📋 Transaction Details</div>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Date & Time</th>
                    <th>Customer</th>
                    <th>Amount</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Payment Method</th>
                </tr>
            </thead>
            <tbody>
                <?php foreach (array_slice($transactions, 0, 100) as $transaction): // Limit to 100 for PDF
                    $amount = abs($transaction['amount']);
                ?>
                <tr>
                    <td>#<?= $transaction['id'] ?></td>
                    <td><?= date('M j, Y H:i', strtotime($transaction['timestamp'])) ?></td>
                    <td>
                        <?= htmlspecialchars($transaction['full_name'] ?? 'Unknown') ?><br>
                        <small style="color: #666;"><?= htmlspecialchars($transaction['email'] ?? '') ?></small>
                    </td>
                    <td class="<?= $transaction['transaction_type'] === 'refund' ? 'negative' : 'positive' ?>">
                        <?= $transaction['transaction_type'] === 'refund' ? '-' : '' ?>RM <?= number_format($amount, 2) ?>
                    </td>
                    <td style="text-transform: capitalize;"><?= htmlspecialchars($transaction['transaction_type']) ?></td>
                    <td>
                        <span style="color: <?= $transaction['status'] === 'success' ? '#27ae60' : ($transaction['status'] === 'failed' ? '#e74c3c' : '#f39c12') ?>">
                            <?= ucfirst($transaction['status']) ?>
                        </span>
                    </td>
                    <td style="text-transform: capitalize;"><?= htmlspecialchars($transaction['payment_method'] ?? 'N/A') ?></td>
                </tr>
                <?php endforeach; ?>
            </tbody>
        </table>
        
        <?php if (count($transactions) > 100): ?>
        <p style="margin-top: 10px; font-style: italic; color: #666;">
            Showing first 100 transactions. For complete data, please contact admin.
        </p>
        <?php endif; ?>

        <?php if ($total_refunds > 0): ?>
        <!-- Refunds Summary -->
        <div style="margin-top: 30px; padding: 15px; background: #fff3cd; border: 1px solid #ffc107; border-radius: 5px;">
            <h4 style="color: #856404; margin-bottom: 10px;">📋 Refunds Summary</h4>
            <p><strong>Total Refunded:</strong> RM <?= number_format($total_refunds, 2) ?></p>
            <p><strong>Refunded Transactions:</strong> <?= count(array_filter($transactions, fn($t) => $t['status'] === 'refunded' || $t['transaction_type'] === 'refund')) ?></p>
            <p style="margin-bottom: 0;"><strong>Net Revenue after Refunds:</strong> RM <?= number_format($net_revenue, 2) ?></p>
        </div>
        <?php endif; ?>

        <div style="margin-top: 40px; padding-top: 20px; border-top: 2px solid #ddd; text-align: center; color: #666; font-size: 12px;">
            <p>This simplified report is generated by FacePay Admin System</p>
            <p>All amounts are in Malaysian Ringgit (MYR)</p>
            <p>For questions about this report, please contact your account manager</p>
        </div>
    </body>
    </html>
    <?php
}
?>