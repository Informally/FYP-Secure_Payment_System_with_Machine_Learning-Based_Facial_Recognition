<?php
// payment_gateway/api/verify_payment.php
// API endpoint to verify payment status

header('Content-Type: application/json');
require_once '../../customer_side/config.php';

function sendResponse($status, $message, $data = null) {
    http_response_code($status);
    echo json_encode([
        'status' => $status < 400 ? 'success' : 'error',
        'message' => $message,
        'data' => $data,
        'timestamp' => time()
    ]);
    exit();
}

if ($_SERVER['REQUEST_METHOD'] !== 'GET') {
    sendResponse(405, 'Method not allowed');
}

// Get Authorization header
$headers = getallheaders();
$authHeader = $headers['Authorization'] ?? '';

if (!preg_match('/^Bearer\s+(.+)$/', $authHeader, $matches)) {
    sendResponse(401, 'Missing or invalid authorization header');
}

$apiKey = $matches[1];

// Validate API key
$conn = dbConnect();
$merchant_stmt = $conn->prepare("SELECT id, merchant_id, name FROM merchants WHERE api_key = ? AND is_active = 1");
$merchant_stmt->bind_param("s", $apiKey);
$merchant_stmt->execute();
$merchant_result = $merchant_stmt->get_result();

if ($merchant_result->num_rows === 0) {
    sendResponse(401, 'Invalid API key');
}

$merchant = $merchant_result->fetch_assoc();

$transaction_id = $_GET['transaction_id'] ?? '';
$order_id = $_GET['order_id'] ?? '';

if (empty($transaction_id) && empty($order_id)) {
    sendResponse(400, 'Either transaction_id or order_id is required');
}

// Query transaction
if ($transaction_id) {
    $stmt = $conn->prepare("
        SELECT t.*, u.full_name, u.user_id 
        FROM transactions t 
        JOIN users u ON t.user_id = u.id 
        WHERE t.id = ? AND t.merchant_id = ?
    ");
    $stmt->bind_param("is", $transaction_id, $merchant['merchant_id']);
} else {
    $stmt = $conn->prepare("
        SELECT t.*, u.full_name, u.user_id 
        FROM transactions t 
        JOIN users u ON t.user_id = u.id 
        WHERE t.order_id = ? AND t.merchant_id = ?
    ");
    $stmt->bind_param("ss", $order_id, $merchant['merchant_id']);
}

$stmt->execute();
$result = $stmt->get_result();

if ($result->num_rows === 0) {
    sendResponse(404, 'Transaction not found');
}

$transaction = $result->fetch_assoc();
$conn->close();

// Return transaction details
sendResponse(200, 'Transaction found', [
    'transaction_id' => $transaction['id'],
    'order_id' => $transaction['order_id'],
    'amount' => abs($transaction['amount']), // Convert back to positive
    'status' => $transaction['status'],
    'payment_method' => $transaction['payment_method'],
    'customer_id' => $transaction['user_id'],
    'customer_name' => $transaction['full_name'],
    'merchant_id' => $transaction['merchant_id'],
    'merchant_name' => $transaction['merchant_name'],
    'timestamp' => $transaction['timestamp'],
    'transaction_type' => $transaction['transaction_type']
]);
?>