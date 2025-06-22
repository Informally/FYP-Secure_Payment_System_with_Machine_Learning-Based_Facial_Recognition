<?php
// export_merchant_report.php
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
$format = $_POST['export_format'] ?? 'pdf';
$include_summary = isset($_POST['include_summary']);
$include_transactions = isset($_POST['include_transactions']);
$include_charts = isset($_POST['include_charts']);
$include_fees = isset($_POST['include_fees']);

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
    
    // Calculate summary statistics
    $total_transactions = count($transactions);
    $successful_transactions = array_filter($transactions, fn($t) => $t['status'] === 'success' && $t['transaction_type'] === 'payment');
    $failed_transactions = array_filter($transactions, fn($t) => $t['status'] === 'failed');
    $refunded_transactions = array_filter($transactions, fn($t) => $t['status'] === 'refunded');
    
    $total_revenue = array_sum(array_map(fn($t) => $t['status'] === 'success' && $t['transaction_type'] === 'payment' ? abs($t['amount']) : 0, $transactions));
    $total_refunds = array_sum(array_map(fn($t) => $t['status'] === 'refunded' || $t['transaction_type'] === 'refund' ? abs($t['amount']) : 0, $transactions));
    $net_revenue = $total_revenue - $total_refunds;
    $platform_fees = $total_revenue * 0.029;
    $merchant_earnings = $total_revenue - $platform_fees;
    
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
    
    // Generate report based on format
    switch ($format) {
        // case 'csv':
        //     generateCSV($merchant, $transactions, $date_from, $date_to);
        //     break;
        // case 'excel':
        //     generateExcel($merchant, $transactions, $daily_data, $total_revenue, $platform_fees, $merchant_earnings, $success_rate, $date_from, $date_to);
        //     break;
        case 'pdf':
        default:
            generatePDF($merchant, $transactions, $daily_data, $total_revenue, $platform_fees, $merchant_earnings, $success_rate, $avg_transaction, $total_refunds, $include_summary, $include_transactions, $include_charts, $include_fees, $date_from, $date_to);
            break;
    }
    
} catch (Exception $e) {
    error_log("Export error: " . $e->getMessage());
    die('Error generating report: ' . $e->getMessage());
}

function generateCSV($merchant, $transactions, $date_from, $date_to) {
    $filename = "merchant_report_{$merchant['merchant_id']}_{$date_from}_to_{$date_to}.csv";
    
    header('Content-Type: text/csv');
    header('Content-Disposition: attachment; filename="' . $filename . '"');
    
    $output = fopen('php://output', 'w');
    
    // CSV Headers
    fputcsv($output, [
        'Transaction ID', 'Date', 'Time', 'Customer Name', 'Customer Email', 
        'Amount (RM)', 'Platform Fee (RM)', 'Net Amount (RM)', 'Status', 'Type', 'Order ID'
    ]);
    
    foreach ($transactions as $transaction) {
        $amount = abs($transaction['amount']);
        $fee = $transaction['status'] === 'success' && $transaction['transaction_type'] === 'payment' ? $amount * 0.029 : 0;
        $net = $amount - $fee;
        
        fputcsv($output, [
            $transaction['id'],
            date('Y-m-d', strtotime($transaction['timestamp'])),
            date('H:i:s', strtotime($transaction['timestamp'])),
            $transaction['full_name'] ?? 'Unknown',
            $transaction['email'] ?? 'N/A',
            number_format($amount, 2),
            number_format($fee, 2),
            number_format($net, 2),
            $transaction['status'],
            $transaction['transaction_type'],
            $transaction['order_id'] ?? 'N/A'
        ]);
    }
    
    fclose($output);
}

function generateExcel($merchant, $transactions, $daily_data, $total_revenue, $platform_fees, $merchant_earnings, $success_rate, $date_from, $date_to) {
    // For Excel, we'll generate a comprehensive CSV with multiple sheets worth of data
    $filename = "merchant_report_{$merchant['merchant_id']}_{$date_from}_to_{$date_to}.csv";
    
    header('Content-Type: application/vnd.ms-excel');
    header('Content-Disposition: attachment; filename="' . $filename . '"');
    
    echo "MERCHANT FINANCIAL REPORT\n";
    echo "Merchant: {$merchant['merchant_name']} ({$merchant['merchant_id']})\n";
    echo "Period: {$date_from} to {$date_to}\n";
    echo "Generated: " . date('Y-m-d H:i:s') . "\n\n";
    
    echo "FINANCIAL SUMMARY\n";
    echo "Total Revenue,RM " . number_format($total_revenue, 2) . "\n";
    echo "Platform Fees (2.9%),RM " . number_format($platform_fees, 2) . "\n";
    echo "Your Earnings,RM " . number_format($merchant_earnings, 2) . "\n";
    echo "Success Rate," . number_format($success_rate, 1) . "%\n";
    echo "Total Transactions," . count($transactions) . "\n\n";
    
    echo "DAILY BREAKDOWN\n";
    echo "Date,Revenue,Transactions,Success Rate\n";
    foreach ($daily_data as $day) {
        $day_success_rate = $day['transaction_count'] > 0 ? ($day['successful_count'] / $day['transaction_count']) * 100 : 0;
        echo "{$day['date']},RM " . number_format($day['revenue'], 2) . ",{$day['transaction_count']}," . number_format($day_success_rate, 1) . "%\n";
    }
    
    echo "\nTRANSACTION DETAILS\n";
    echo "ID,Date,Time,Customer,Amount,Platform Fee,Your Earning,Status,Type\n";
    
    foreach ($transactions as $transaction) {
        $amount = abs($transaction['amount']);
        $fee = $transaction['status'] === 'success' && $transaction['transaction_type'] === 'payment' ? $amount * 0.029 : 0;
        $earning = $amount - $fee;
        
        echo "{$transaction['id']}," .
             date('Y-m-d', strtotime($transaction['timestamp'])) . "," .
             date('H:i:s', strtotime($transaction['timestamp'])) . "," .
             "\"{$transaction['full_name']}\"," .
             number_format($amount, 2) . "," .
             number_format($fee, 2) . "," .
             number_format($earning, 2) . "," .
             "{$transaction['status']}," .
             "{$transaction['transaction_type']}\n";
    }
}

function generatePDF($merchant, $transactions, $daily_data, $total_revenue, $platform_fees, $merchant_earnings, $success_rate, $avg_transaction, $total_refunds, $include_summary, $include_transactions, $include_charts, $include_fees, $date_from, $date_to) {
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
            .fees { border-left-color: #ed8936; }
            .earnings { border-left-color: #3182ce; }
            .success { border-left-color: #38a169; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background: #667eea; color: white; }
            tr:nth-child(even) { background: #f9f9f9; }
            .section-title { font-size: 20px; color: #667eea; margin: 30px 0 15px 0; border-bottom: 2px solid #667eea; padding-bottom: 5px; }
            .fee-breakdown { background: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 5px; margin: 20px 0; }
            .highlight { color: #e74c3c; font-weight: bold; }
            .positive { color: #27ae60; font-weight: bold; }
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

        <?php if ($include_summary): ?>
        <div class="section-title">📊 Financial Summary</div>
        <div class="summary-grid">
            <div class="summary-card revenue">
                <div class="card-title">Total Revenue</div>
                <div class="card-value">RM <?= number_format($total_revenue, 2) ?></div>
            </div>
            <div class="summary-card fees">
                <div class="card-title">Platform Fees (2.9%)</div>
                <div class="card-value">RM <?= number_format($platform_fees, 2) ?></div>
            </div>
            <div class="summary-card earnings">
                <div class="card-title">Your Net Earnings</div>
                <div class="card-value positive">RM <?= number_format($merchant_earnings, 2) ?></div>
            </div>
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
        <?php endif; ?>

        <?php if ($include_fees): ?>
        <div class="section-title">💰 Platform Fees Breakdown</div>
        <div class="fee-breakdown">
            <h4>How Platform Fees Are Calculated:</h4>
            <p><strong>Rate:</strong> 2.9% per successful transaction</p>
            <p><strong>Total Revenue:</strong> RM <?= number_format($total_revenue, 2) ?></p>
            <p><strong>Platform Fees:</strong> RM <?= number_format($total_revenue, 2) ?> × 2.9% = <span class="highlight">RM <?= number_format($platform_fees, 2) ?></span></p>
            <p><strong>Your Earnings:</strong> RM <?= number_format($total_revenue, 2) ?> - RM <?= number_format($platform_fees, 2) ?> = <span class="positive">RM <?= number_format($merchant_earnings, 2) ?></span></p>
            
            <?php if ($total_refunds > 0): ?>
            <hr style="margin: 15px 0;">
            <p><strong>Note:</strong> Refunds totaling RM <?= number_format($total_refunds, 2) ?> have been processed. Platform fees are not charged on refunded transactions.</p>
            <?php endif; ?>
        </div>
        <?php endif; ?>

        <?php if ($include_charts && !empty($daily_data)): ?>
        <div class="section-title">📈 Daily Performance</div>
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Revenue</th>
                    <th>Platform Fee</th>
                    <th>Your Earnings</th>
                    <th>Transactions</th>
                    <th>Success Rate</th>
                </tr>
            </thead>
            <tbody>
                <?php foreach ($daily_data as $day): 
                    $day_fee = $day['revenue'] * 0.029;
                    $day_earnings = $day['revenue'] - $day_fee;
                    $day_success_rate = $day['transaction_count'] > 0 ? ($day['successful_count'] / $day['transaction_count']) * 100 : 0;
                ?>
                <tr>
                    <td><?= date('M j, Y', strtotime($day['date'])) ?></td>
                    <td>RM <?= number_format($day['revenue'], 2) ?></td>
                    <td>RM <?= number_format($day_fee, 2) ?></td>
                    <td class="positive">RM <?= number_format($day_earnings, 2) ?></td>
                    <td><?= $day['transaction_count'] ?></td>
                    <td><?= number_format($day_success_rate, 1) ?>%</td>
                </tr>
                <?php endforeach; ?>
            </tbody>
        </table>
        <?php endif; ?>

        <?php if ($include_transactions): ?>
        <div class="section-title">📋 Transaction Details</div>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Date & Time</th>
                    <th>Customer</th>
                    <th>Amount</th>
                    <th>Platform Fee</th>
                    <th>Your Earning</th>
                    <th>Status</th>
                    <th>Type</th>
                </tr>
            </thead>
            <tbody>
                <?php foreach (array_slice($transactions, 0, 100) as $transaction): // Limit to 100 for PDF
                    $amount = abs($transaction['amount']);
                    $fee = $transaction['status'] === 'success' && $transaction['transaction_type'] === 'payment' ? $amount * 0.029 : 0;
                    $earning = $amount - $fee;
                ?>
                <tr>
                    <td>#<?= $transaction['id'] ?></td>
                    <td><?= date('M j, Y H:i', strtotime($transaction['timestamp'])) ?></td>
                    <td><?= htmlspecialchars($transaction['full_name'] ?? 'Unknown') ?></td>
                    <td>RM <?= number_format($amount, 2) ?></td>
                    <td>RM <?= number_format($fee, 2) ?></td>
                    <td class="positive">RM <?= number_format($earning, 2) ?></td>
                    <td>
                        <span style="color: <?= $transaction['status'] === 'success' ? '#27ae60' : ($transaction['status'] === 'failed' ? '#e74c3c' : '#f39c12') ?>">
                            <?= ucfirst($transaction['status']) ?>
                        </span>
                    </td>
                    <td><?= ucfirst($transaction['transaction_type']) ?></td>
                </tr>
                <?php endforeach; ?>
            </tbody>
        </table>
        
        <?php if (count($transactions) > 100): ?>
        <p style="margin-top: 10px; font-style: italic; color: #666;">
            Showing first 100 transactions. For complete data, please export as Excel or CSV format.
        </p>
        <?php endif; ?>
        <?php endif; ?>

        <div style="margin-top: 40px; padding-top: 20px; border-top: 2px solid #ddd; text-align: center; color: #666; font-size: 12px;">
            <p>This report is generated by FacePay Admin System</p>
            <p>For questions about this report, please contact your account manager</p>
        </div>
    </body>
    </html>
    <?php
}
?>