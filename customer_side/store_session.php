<?php
// Updated store_session.php - Enhanced with security logging (MySQLi version)
session_start();
require_once 'config.php';

header('Content-Type: application/json');

// Security headers
header('X-Content-Type-Options: nosniff');
header('X-Frame-Options: DENY');
header('X-XSS-Protection: 1; mode=block');

// Only allow POST requests
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['status' => 'error', 'message' => 'Method not allowed']);
    exit();
}

// Rate limiting
$client_ip = $_SERVER['REMOTE_ADDR'] ?? 'unknown';
if (!SecurityUtils::checkRateLimit('session_' . $client_ip, 20, 300)) { // 20 requests per 5 minutes
    http_response_code(429);
    echo json_encode(['status' => 'error', 'message' => 'Too many requests']);
    exit();
}

// Get JSON input
$input = file_get_contents('php://input');
$data = json_decode($input, true);

if (!$data) {
    echo json_encode(['status' => 'error', 'message' => 'Invalid JSON data']);
    exit();
}

// Handle different actions
switch ($data['action'] ?? '') {
    case 'store_authenticated_user':
        if (isset($data['user_id'])) {
            // Verify user exists in database
            try {
                $conn = dbConnect(); // Use MySQLi connection
                
                // MySQLi prepared statement
                $stmt = $conn->prepare("SELECT id, username, account_status FROM users WHERE user_id = ?");
                $stmt->bind_param("s", $data['user_id']);
                $stmt->execute();
                $result = $stmt->get_result();
                $user = $result->fetch_assoc();
                
                if (!$user) {
                    SecurityUtils::logSecurityEvent($data['user_id'], 'invalid_user_session', 'failure');
                    echo json_encode(['status' => 'error', 'message' => 'Invalid user']);
                    $stmt->close();
                    $conn->close();
                    exit();
                }
                
                if ($user['account_status'] !== 'active') {
                    SecurityUtils::logSecurityEvent($data['user_id'], 'inactive_user_session', 'failure');
                    echo json_encode(['status' => 'error', 'message' => 'Account not active']);
                    $stmt->close();
                    $conn->close();
                    exit();
                }
                
                // Store session data
                $_SESSION['authenticatedUserId'] = $data['user_id'];
                $_SESSION['username'] = $user['username'];
                $_SESSION['auth_level'] = 'basic';
                $_SESSION['auth_time'] = time();
                
                // Store cart data if provided
                if (isset($data['cart'])) {
                    $_SESSION['kioskCart'] = $data['cart'];
                }
                
                // Store authentication metadata
                if (isset($data['liveness_score'])) {
                    $_SESSION['liveness_score'] = $data['liveness_score'];
                }
                if (isset($data['similarity'])) {
                    $_SESSION['similarity'] = $data['similarity'];
                }
                
                SecurityUtils::logSecurityEvent($data['user_id'], 'kiosk_session_created', 'success', [
                    'liveness_score' => $data['liveness_score'] ?? null,
                    'similarity' => $data['similarity'] ?? null
                ]);
                
                echo json_encode([
                    'status' => 'success', 
                    'message' => 'User authenticated and stored in session',
                    'user_id' => $data['user_id'],
                    'username' => $user['username']
                ]);
                
                $stmt->close();
                $conn->close();
                
            } catch (Exception $e) {
                error_log("Session storage error: " . $e->getMessage());
                echo json_encode(['status' => 'error', 'message' => 'Database error']);
            }
        } else {
            echo json_encode(['status' => 'error', 'message' => 'User ID not provided']);
        }
        break;
        
    case 'clear_session':
        $user_id = $_SESSION['authenticatedUserId'] ?? 'unknown';
        
        // Clear authentication data
        unset($_SESSION['authenticatedUserId']);
        unset($_SESSION['username']);
        unset($_SESSION['auth_level']);
        unset($_SESSION['liveness_score']);
        unset($_SESSION['similarity']);
        unset($_SESSION['kioskCart']);
        
        SecurityUtils::logSecurityEvent($user_id, 'session_cleared', 'success');
        
        echo json_encode(['status' => 'success', 'message' => 'Session cleared']);
        break;
        
    case 'check_session':
        // Check if user is authenticated
        $isAuthenticated = isset($_SESSION['authenticatedUserId']);
        
        // Check session timeout
        if ($isAuthenticated && isset($_SESSION['auth_time'])) {
            if (time() - $_SESSION['auth_time'] > 1800) { // 30 minutes
                // Session expired
                unset($_SESSION['authenticatedUserId']);
                unset($_SESSION['username']);
                unset($_SESSION['auth_level']);
                $isAuthenticated = false;
                
                SecurityUtils::logSecurityEvent($_SESSION['authenticatedUserId'] ?? 'unknown', 'session_expired', 'success');
            }
        }
        
        echo json_encode([
            'status' => 'success',
            'authenticated' => $isAuthenticated,
            'user_id' => $_SESSION['authenticatedUserId'] ?? null,
            'username' => $_SESSION['username'] ?? null,
            'auth_level' => $_SESSION['auth_level'] ?? null,
            'session_time_remaining' => $isAuthenticated ? (1800 - (time() - $_SESSION['auth_time'])) : 0
        ]);
        break;
        
    case 'extend_session':
        // Extend session for active users
        if (isset($_SESSION['authenticatedUserId'])) {
            $_SESSION['auth_time'] = time();
            echo json_encode(['status' => 'success', 'message' => 'Session extended']);
        } else {
            echo json_encode(['status' => 'error', 'message' => 'No active session']);
        }
        break;
        
    default:
        echo json_encode(['status' => 'error', 'message' => 'Unknown action']);
        break;
}
?>