<?php
require_once '../config.php';
require_once '../encrypt.php';
header('Content-Type: application/json');

$method = $_SERVER['REQUEST_METHOD'];
$action = isset($_GET['action']) ? $_GET['action'] : '';

// Rate limiting
$rate_limit_key = 'rate_limit_' . $_SERVER['REMOTE_ADDR'];
$rate_limit_window = 60; // 1 minute
$max_requests = 10;

if (!isset($_SESSION[$rate_limit_key])) {
    $_SESSION[$rate_limit_key] = ['count' => 0, 'start_time' => time()];
}

if (time() - $_SESSION[$rate_limit_key]['start_time'] > $rate_limit_window) {
    $_SESSION[$rate_limit_key] = ['count' => 0, 'start_time' => time()];
}

if ($_SESSION[$rate_limit_key]['count'] >= $max_requests) {
    http_response_code(429);
    echo json_encode(['status' => 'error', 'message' => 'Too many requests. Please try again later.']);
    exit;
}

$_SESSION[$rate_limit_key]['count']++;

if ($method === 'POST') {
    $json_data = file_get_contents('php://input');
    $data = json_decode($json_data, true);
    
    if (!$data) {
        http_response_code(400);
        echo json_encode(['status' => 'error', 'message' => 'Invalid JSON data']);
        exit;
    }
    
    if (empty($action) && isset($data['action'])) {
        $action = $data['action'];
    }
    
    // Validate CSRF token for sensitive actions
    if (in_array($action, ['process_payment', 'process_order'])) {
        $csrf_token = $data['csrf_token'] ?? '';
        if (!validateCSRFToken($csrf_token)) {
            http_response_code(403);
            echo json_encode(['status' => 'error', 'message' => 'Invalid CSRF token']);
            exit;
        }
    }
    
    switch ($action) {
        case 'verify_face':
            handleFaceVerification($data);
            break;
        case 'verify_pin':
            handlePinVerification($data);
            break;
        case 'process_payment':
            handlePaymentProcessing($data);
            break;
        case 'check_balance':
            handleBalanceCheck($data);
            break;
        case 'process_order':
            handleOrderProcessing($data);
            break;
        default:
            http_response_code(404);
            echo json_encode(['status' => 'error', 'message' => 'Unknown action']);
    }
} else {
    http_response_code(405);
    echo json_encode(['status' => 'error', 'message' => 'Method not allowed']);
}

function handleFaceVerification($data) {
    if (!isset($data['image_base64'])) {
        http_response_code(400);
        echo json_encode(['status' => 'error', 'message' => 'Missing face image data']);
        return;
    }
    
    $image_data = $data['image_base64'];
    $amount = $data['amount'] ?? 0; // Pass transaction amount for adaptive thresholds
    
    $api_url = "http://localhost:5000/authenticate";
    $api_data = [
        'image_base64' => $image_data,
        'challenge_type' => 'head_movement',
        'additional_frames' => isset($data['additional_frames']) ? $data['additional_frames'] : [],
        'transaction_amount' => $amount
    ];
    
    $ch = curl_init($api_url);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($api_data));
    curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
    
    $response = curl_exec($ch);
    
    if (curl_errno($ch)) {
        http_response_code(500);
        echo json_encode(['status' => 'error', 'message' => 'Failed to connect to face verification service']);
        curl_close($ch);
        return;
    }
    
    $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);
    
    if ($http_code !== 200) {
        http_response_code(500);
        echo json_encode(['status' => 'error', 'message' => 'Face verification service returned an error']);
        return;
    }
    
    $result = json_decode($response, true);
    
    if (!$result) {
        http_response_code(500);
        echo json_encode(['status' => 'error', 'message' => 'Invalid response from face verification service']);
        return;
    }
    
    if ($result['status'] === 'success' && $result['liveness_verified']) {
        echo json_encode([
            'status' => 'success', 
            'user_id' => $result['user_id'],
            'message' => 'Face verification successful',
            'liveness_score' => $result['liveness_score'] ?? 0.95,
            'similarity' => $result['similarity'] ?? 0.92
        ]);
    } else {
        echo json_encode([
            'status' => 'error', 
            'message' => $result['message'] ?? 'Face verification failed'
        ]);
    }
}

function handlePinVerification($data) {
    if (!isset($data['user_id']) || !isset($data['pin'])) {
        http_response_code(400);
        echo json_encode(['status' => 'error', 'message' => 'Missing user_id or PIN']);
        return;
    }
    
    $user_id = $data['user_id'];
    $pin = $data['pin'];
    
    $conn = dbConnect();
    
    $stmt = $conn->prepare("SELECT pin_code FROM users WHERE id = ?");
    $stmt->bind_param("i", $user_id);
    $stmt->execute();
    $result = $stmt->get_result();
    
    if ($result->num_rows === 0) {
        http_response_code(404);
        echo json_encode(['status' => 'error', 'message' => 'User not found']);
        $conn->close();
        return;
    }
    
    $row = $result->fetch_assoc();
    $encrypted_pin = $row['pin_code'];
    
    $decrypted_pin = aes_decrypt($encrypted_pin);
    
    if ($pin === $decrypted_pin) {
        echo json_encode(['status' => 'success', 'message' => 'PIN verified successfully']);
    } else {
        echo json_encode(['status' => 'error', 'message' => 'Incorrect PIN']);
    }
    
    $conn->close();
}

function handleBalanceCheck($data) {
    if (!isset($data['user_id'])) {
        http_response_code(400);
        echo json_encode(['status' => 'error', 'message' => 'Missing user_id']);
        return;
    }
    
    $user_id = $data['user_id'];
    
    $conn = dbConnect();
    
    $stmt = $conn->prepare("SELECT balance FROM wallets WHERE user_id = ?");
    $stmt->bind_param("i", $user_id);
    $stmt->execute();
    $result = $stmt->get_result();
    
    if ($result->num_rows === 0) {
        http_response_code(404);
        echo json_encode(['status' => 'error', 'message' => 'Wallet not found']);
        $conn->close();
        return;
    }
    
    $row = $result->fetch_assoc();
    $balance = floatval($row['balance']);
    
    echo json_encode([
        'status' => 'success', 
        'balance' => $balance,
        'formatted_balance' => number_format($balance, 2)
    ]);
    
    $conn->close();
}

function handlePaymentProcessing($data) {
    if (!isset($data['user_id']) || !isset($data['amount'])) {
        http_response_code(400);
        echo json_encode(['status' => 'error', 'message' => 'Missing required fields']);
        return;
    }
    
    $user_id = $data['user_id'];
    $amount = floatval($data['amount']);
    $merchant_id = $data['merchant_id'] ?? null;
    $merchant_name = $data['merchant_name'] ?? 'Unknown Merchant';
    $order_id = $data['order_id'] ?? null;
    $payment_method = $data['payment_method'] ?? 'facial';
    
    if ($amount <= 0) {
        http_response_code(400);
        echo json_encode(['status' => 'error', 'message' => 'Invalid amount']);
        return;
    }
    
    $conn = dbConnect();
    
    $conn->begin_transaction();
    
    try {
        $stmt = $conn->prepare("SELECT balance FROM wallets WHERE user_id = ? FOR UPDATE");
        $stmt->bind_param("i", $user_id);
        $stmt->execute();
        $result = $stmt->get_result();
        
        if ($result->num_rows === 0) {
            throw new Exception('Wallet not found');
        }
        
        $row = $result->fetch_assoc();
        $balance = floatval($row['balance']);
        
        if ($balance < $amount) {
            $conn->rollback();
            echo json_encode([
                'status' => 'error', 
                'message' => 'Insufficient balance',
                'current_balance' => $balance,
                'required_amount' => $amount
            ]);
            $conn->close();
            return;
        }
        
        $stmt = $conn->prepare("UPDATE wallets SET balance = balance - ? WHERE user_id = ?");
        $stmt->bind_param("di", $amount, $user_id);
        $stmt->execute();
        
        if ($stmt->affected_rows !== 1) {
            throw new Exception('Failed to update wallet balance');
        }
        
        $transaction_id = 'TXN-' . strtoupper(substr(md5(uniqid(mt_rand(), true)), 0, 10));
        
        $stmt = $conn->prepare("INSERT INTO transactions (user_id, amount, status, merchant_id, merchant_name, payment_method, order_id) VALUES (?, ?, 'success', ?, ?, ?, ?)");
        $stmt->bind_param("idssss", $user_id, $amount, $merchant_id, $merchant_name, $payment_method, $order_id);
        $stmt->execute();
        
        if ($stmt->affected_rows !== 1) {
            throw new Exception('Failed to record transaction');
        }
        
        $transaction_id = 'TXN-' . $stmt->insert_id;
        
        if ($merchant_id) {
            $stmt = $conn->prepare("UPDATE merchants SET balance = balance + ? WHERE merchant_id = ?");
            $stmt->bind_param("ds", $amount, $merchant_id);
            $stmt->execute();
        }
        
        if ($order_id) {
            $stmt = $conn->prepare("UPDATE orders SET status = 'completed' WHERE order_id = ?");
            $stmt->bind_param("s", $order_id);
            $stmt->execute();
        }
        
        $stmt = $conn->prepare("INSERT INTO transaction_history (transaction_id, order_id, user_id, merchant_id, type, amount, status, payment_method) VALUES (?, ?, ?, ?, 'payment', ?, 'success', ?)");
        $stmt->bind_param("ssisss", $transaction_id, $order_id, $user_id, $merchant_id, $amount, $payment_method);
        $stmt->execute();
        
        $conn->commit();
        
        echo json_encode([
            'status' => 'success',
            'message' => 'Payment processed successfully',
            'transaction_id' => $transaction_id,
            'amount' => $amount,
            'new_balance' => $balance - $amount,
            'formatted_new_balance' => number_format($balance - $amount, 2)
        ]);
        
    } catch (Exception $e) {
        $conn->rollback();
        http_response_code(500);
        echo json_encode(['status' => 'error', 'message' => $e->getMessage()]);
    }
    
    $conn->close();
}

function handleOrderProcessing($data) {
    if (!isset($data['user_id']) || !isset($data['items']) || !isset($data['total_amount'])) {
        http_response_code(400);
        echo json_encode(['status' => 'error', 'message' => 'Missing required order data']);
        return;
    }
    
    $user_id = $data['user_id'];
    $items = $data['items'];
    $total_amount = floatval($data['total_amount']);
    $merchant_id = $data['merchant_id'] ?? null;
    $merchant_name = $data['merchant_name'] ?? 'Unknown Merchant';
    $shipping_info = $data['shipping_info'] ?? null;
    
    $order_id = 'ORD-' . strtoupper(substr(md5(uniqid(mt_rand(), true)), 0, 8));
    
    $conn = dbConnect();
    
    $conn->begin_transaction();
    
    try {
        $stmt = $conn->prepare("INSERT INTO orders (order_id, user_id, merchant_id, merchant_name, total_amount, status) VALUES (?, ?, ?, ?, ?, 'pending')");
        $stmt->bind_param("sissd", $order_id, $user_id, $merchant_id, $merchant_name, $total_amount);
        $stmt->execute();
        
        if ($stmt->affected_rows !== 1) {
            throw new Exception('Failed to create order');
        }
        
        $stmt = $conn->prepare("INSERT INTO order_items (order_id, product_id, product_name, quantity, price, subtotal) VALUES (?, ?, ?, ?, ?, ?)");
        
        foreach ($items as $item) {
            $product_id = $item['id'];
            $product_name = $item['name'];
            $quantity = $item['quantity'];
            $price = $item['price'];
            $subtotal = $price * $quantity;
            
            $stmt->bind_param("sissdd", $order_id, $product_id, $product_name, $quantity, $price, $subtotal);
            $stmt->execute();
            
            if ($stmt->affected_rows !== 1) {
                throw new Exception('Failed to add order item');
            }
            
            $update_stock = $conn->prepare("UPDATE products SET stock = stock - ? WHERE id = ?");
            $update_stock->bind_param("ii", $quantity, $product_id);
            $update_stock->execute();
        }
        
        if ($shipping_info) {
            $stmt = $conn->prepare("INSERT INTO shipping_info 
                (user_id, order_id, full_name, address_line1, address_line2, city, state, postal_code, country, phone) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
                
            $stmt->bind_param("isssssssss", 
                $user_id, 
                $order_id, 
                $shipping_info['full_name'], 
                $shipping_info['address_line1'], 
                $shipping_info['address_line2'] ?? null, 
                $shipping_info['city'], 
                $shipping_info['state'], 
                $shipping_info['postal_code'], 
                $shipping_info['country'], 
                $shipping_info['phone']
            );
            
            $stmt->execute();
        }
        
        $conn->commit();
        
        echo json_encode([
            'status' => 'success',
            'message' => 'Order created successfully',
            'order_id' => $order_id,
            'total_amount' => $total_amount
        ]);
        
    } catch (Exception $e) {
        $conn->rollback();
        http_response_code(500);
        echo json_encode(['status' => 'error', 'message' => $e->getMessage()]);
    }
    
    $conn->close();
}
?>