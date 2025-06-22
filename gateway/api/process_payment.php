<?php
// payment_gateway/api/process_payment.php
// API endpoint for direct payment processing (like Stripe's API)

header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type, Authorization');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    exit(0);
}

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

// Validate request method
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    sendResponse(405, 'Method not allowed');
}

// Get Authorization header
$headers = getallheaders();
$authHeader = $headers['Authorization'] ?? '';

if (!preg_match('/^Bearer\s+(.+)$/', $authHeader, $matches)) {
    sendResponse(401, 'Missing or invalid authorization header');
}

$apiKey = $matches[1];

// Validate API key and get merchant
$conn = dbConnect();
$merchant_stmt = $conn->prepare("SELECT id, merchant_id, name, is_active FROM merchants WHERE api_key = ?");
$merchant_stmt->bind_param("s", $apiKey);
$merchant_stmt->execute();
$merchant_result = $merchant_stmt->get_result();

if ($merchant_result->num_rows === 0) {
    sendResponse(401, 'Invalid API key');
}

$merchant = $merchant_result->fetch_assoc();

if (!$merchant['is_active']) {
    sendResponse(403, 'Merchant account is not active');
}

// Parse request body
$input = json_decode(file_get_contents('php://input'), true);
if (!$input) {
    sendResponse(400, 'Invalid JSON payload');
}

// Validate required fields
$required_fields = ['amount', 'order_id', 'customer_id'];
foreach ($required_fields as $field) {
    if (empty($input[$field])) {
        sendResponse(400, "Missing required field: $field");
    }
}

$amount = floatval($input['amount']);
$order_id = $input['order_id'];
$customer_id = $input['customer_id'];
$description = $input['description'] ?? '';
$currency = $input['currency'] ?? 'MYR';

// Validate amount
if ($amount <= 0 || $amount > 10000) {
    sendResponse(400, 'Invalid payment amount');
}

// Check if customer exists and get their details
$customer_stmt = $conn->prepare("SELECT id, full_name, pin_code FROM users WHERE user_id = ?");
$customer_stmt->bind_param("s", $customer_id);
$customer_stmt->execute();
$customer_result = $customer_stmt->get_result();

if ($customer_result->num_rows === 0) {
    sendResponse(404, 'Customer not found');
}

$customer = $customer_result->fetch_assoc();

// Get customer's wallet balance
$wallet_stmt = $conn->prepare("SELECT balance FROM wallets WHERE user_id = ?");
$wallet_stmt->bind_param("i", $customer['id']);
$wallet_stmt->execute();
$wallet_result = $wallet_stmt->get_result();

if ($wallet_result->num_rows === 0) {
    // Create wallet if doesn't exist
    $create_wallet_stmt = $conn->prepare("INSERT INTO wallets (user_id, balance) VALUES (?, 0.00)");
    $create_wallet_stmt->bind_param("i", $customer['id']);
    $create_wallet_stmt->execute();
    $current_balance = 0.00;
} else {
    $wallet_data = $wallet_result->fetch_assoc();
    $current_balance = $wallet_data['balance'];
}

// Check sufficient balance
if ($current_balance < $amount) {
    sendResponse(402, 'Insufficient balance', [
        'required' => $amount,
        'available' => $current_balance
    ]);
}

// For API calls, we'll need the customer to provide their PIN
// or use a pre-authorized token system
$pin_provided = $input['pin'] ?? '';
if (empty($pin_provided)) {
    sendResponse(400, 'PIN required for payment authorization');
}

// Verify PIN
if (aes_decrypt($customer['pin_code']) !== $pin_provided) {
    // Log failed attempt
    $failed_stmt = $conn->prepare("
        INSERT INTO transactions (
            user_id, amount, status, payment_method, timestamp, 
            order_id, merchant_id, merchant_name, transaction_type
        ) VALUES (?, ?, 'failed', 'api', NOW(), ?, ?, ?, 'payment')
    ");
    $negative_amount = -$amount;
    $failed_stmt->bind_param("idsss", 
        $customer['id'], $negative_amount, $order_id, 
        $merchant['merchant_id'], $merchant['name']
    );
    $failed_stmt->execute();
    
    sendResponse(401, 'Invalid PIN');
}

// Process payment
$new_balance = $current_balance - $amount;

// Update wallet
$encrypted_balance = aes_encrypt($new_balance);
$update_stmt = $conn->prepare("UPDATE wallets SET balance = ?, balance_encrypted = ? WHERE user_id = ?");
$update_stmt->bind_param("dsi", $new_balance, $encrypted_balance, $customer['id']);

if (!$update_stmt->execute()) {
    sendResponse(500, 'Payment processing failed');
}

// Record successful transaction
$transaction_stmt = $conn->prepare("
    INSERT INTO transactions (
        user_id, amount, status, payment_method, timestamp,
        order_id, merchant_id, merchant_name, transaction_type
    ) VALUES (?, ?, 'success', 'api', NOW(), ?, ?, ?, 'payment')
");
$negative_amount = -$amount;
$transaction_stmt->bind_param("idsss",
    $customer['id'], $negative_amount, $order_id,
    $merchant['merchant_id'], $merchant['name']
);
$transaction_stmt->execute();

$transaction_id = $conn->insert_id;

$conn->close();

// Return success response
sendResponse(200, 'Payment processed successfully', [
    'transaction_id' => $transaction_id,
    'order_id' => $order_id,
    'amount' => $amount,
    'currency' => $currency,
    'customer_id' => $customer_id,
    'customer_name' => $customer['full_name'],
    'new_balance' => $new_balance,
    'merchant_id' => $merchant['merchant_id'],
    'merchant_name' => $merchant['name'],
    'payment_method' => 'api',
    'timestamp' => date('c')
]);
?>
