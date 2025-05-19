<?php
session_start();
header('Content-Type: application/json');

$data = json_decode(file_get_contents("php://input"), true);

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($data['user_id'])) {
    $_SESSION['user_id'] = $data['user_id'];
    $_SESSION['verified'] = true;
    
    // Store additional metrics if available
    if (isset($data['liveness_score'])) {
        $_SESSION['liveness_score'] = $data['liveness_score'];
    }
    
    if (isset($data['similarity'])) {
        $_SESSION['similarity'] = $data['similarity'];
    }
    
    echo json_encode(['status' => 'success', 'message' => 'Session stored successfully']);
} else {
    http_response_code(400);
    echo json_encode(['status' => 'failed', 'message' => 'Invalid request']);
}
?>