<?php
// customer_side/payment/payment_scan.php - Face Recognition without Rate Limiting
require_once '../config.php';
require_once '../encrypt.php';

// Start session and check for payment session
SessionManager::start();

// Check if we have a valid payment session from gateway
if (!isset($_SESSION['gateway_payment'])) {
    header("Location: ../../gateway/checkout.php");
    exit();
}

// ✅ FLOW PROTECTION - Prevent going back to face scan after completion
if (isset($_SESSION['payment_user']) && $_SESSION['payment_user']['face_verified']) {
    // Face verification already completed - redirect to PIN entry
    header("Location: payment_pin.php");
    exit();
}

$payment_data = $_SESSION['gateway_payment'];

// ✅ ADD THIS AFTER THE EXISTING SESSION VALIDATION
// Get merchant main URL for cancel redirects
$merchant_main_url = '';
if (isset($_SESSION['gateway_payment']['merchant_id'])) {
    try {
        $conn = dbConnect();
        $stmt = $conn->prepare("SELECT return_url FROM merchants WHERE merchant_id = ?");
        $stmt->bind_param("s", $_SESSION['gateway_payment']['merchant_id']);
        $stmt->execute();
        $result = $stmt->get_result();
        $merchant = $result->fetch_assoc();
        $merchant_main_url = $merchant['return_url'] ?? '';
        $stmt->close();
        $conn->close();
    } catch (Exception $e) {
        error_log("Error getting merchant URL: " . $e->getMessage());
    }
}
// Add to payment_data for JavaScript access
$payment_data['merchant_main_url'] = $merchant_main_url;

$error = "";
$success = "";

// Handle facial recognition result for payment
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $action = $_POST['action'] ?? '';
    
    if ($action === 'verify_payment_face') {
        // CSRF validation
        if (!validateCSRFToken($_POST['csrf_token'] ?? '')) {
            $error = "Invalid request. Please try again.";
        } else {
            // Get data from JavaScript
            $user_id = $_POST['user_id'] ?? '';
            $similarity = floatval($_POST['similarity'] ?? 0);
            $liveness_score = floatval($_POST['liveness_score'] ?? 0);
            $liveness_verified = ($_POST['liveness_verified'] ?? 'false') === 'true';
            
            if (empty($user_id)) {
                $error = "No user ID received from facial recognition.";
            } elseif (!$liveness_verified) {
                $error = "Liveness verification failed. Please ensure you're moving your face naturally.";
                SecurityUtils::logSecurityEvent($user_id, 'payment_liveness_failed', 'failure', [
                    'liveness_score' => $liveness_score,
                    'similarity' => $similarity,
                    'payment_amount' => $payment_data['amount'],
                    'merchant_id' => $payment_data['merchant_id']
                ]);
            } elseif ($similarity < Config::FACE_CONFIDENCE_THRESHOLD) {
                $error = "Face recognition confidence too low for payment. Please try again.";
                SecurityUtils::logSecurityEvent($user_id, 'payment_low_confidence', 'failure', [
                    'similarity' => $similarity,
                    'threshold' => Config::FACE_CONFIDENCE_THRESHOLD,
                    'payment_amount' => $payment_data['amount'],
                    'merchant_id' => $payment_data['merchant_id']
                ]);
            } else {
                try {
                    $conn = dbConnect();
                    
                    // Get user details and verify account status
                    $stmt = $conn->prepare("
                        SELECT id, user_id, username, account_status, is_verified, full_name
                        FROM users 
                        WHERE user_id = ?
                    ");
                    $stmt->bind_param("s", $user_id);
                    $stmt->execute();
                    $result = $stmt->get_result();
                    $user = $result->fetch_assoc();
                    $stmt->close();
                    
                    if ($user !== false && $user['account_status'] === 'active' && $user['is_verified']) {
                        // ✅ FLOW PROTECTION - Store payment user session with timestamp
                        $_SESSION['payment_user'] = [
                            'id' => $user['id'],
                            'user_id' => $user['user_id'],
                            'username' => $user['username'],
                            'full_name' => $user['full_name'],
                            'face_verified' => true,
                            'face_similarity' => $similarity,
                            'liveness_score' => $liveness_score,
                            'face_verified_at' => time(), // ✅ Add timestamp
                            'payment_step' => 'pin_entry' // ✅ Track current step
                        ];
                        // ✅ Reset PIN attempts for new payment session
                        unset($_SESSION['payment_pin_attempts']);
                        
                        SecurityUtils::logSecurityEvent($user['user_id'], 'payment_face_success', 'success', [
                            'similarity' => $similarity,
                            'liveness_score' => $liveness_score,
                            'payment_amount' => $payment_data['amount'],
                            'merchant_id' => $payment_data['merchant_id'],
                            'order_id' => $payment_data['order_id']
                        ]);
                        
                        $success = "Face verified! Redirecting to PIN entry...";
                        
                        // JavaScript will handle the redirect to PIN entry
                    } else {
                        $error = "Account not found, not active, or not verified.";
                        SecurityUtils::logSecurityEvent($user_id, 'payment_face_inactive_account', 'failure', [
                            'payment_amount' => $payment_data['amount'],
                            'merchant_id' => $payment_data['merchant_id']
                        ]);
                        
                        // Redirect to merchant
                        echo "<script>
                            setTimeout(() => {
                                window.location.href = '{$payment_data['cancel_url']}?status=failed&reason=account_error&order_id=" . urlencode($payment_data['order_id']) . "';
                            }, 3000);
                        </script>";
                    }
                    
                    $conn->close();
                    
                } catch (Exception $e) {
                    error_log("Payment facial verification error: " . $e->getMessage());
                    $error = "Payment verification failed due to system error.";
                    
                    echo "<script>
                        setTimeout(() => {
                            window.location.href = '{$payment_data['cancel_url']}?status=failed&reason=system_error&order_id=" . urlencode($payment_data['order_id']) . "';
                        }, 3000);
                    </script>";
                }
            }
        }
    }
}

$csrf_token = generateCSRFToken();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Payment Authorization - FacePay</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --primary: #667eea;
            --primary-light: #e3f2fd;
            --primary-dark: #5a67d8;
            --success: #38a169;
            --warning: #ed8936;
            --error: #e53e3e;
            --text-dark: #2d3748;
            --text-medium: #4a5568;
            --text-light: #718096;
            --border: #e2e8f0;
            --bg-light: #f7fafc;
            --shadow: rgba(0,0,0,0.1);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: var(--text-dark);
            line-height: 1.6;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            width: 100%;
            max-width: 900px;
        }
        
        .payment-container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            text-align: center;
            animation: fadeUp 0.6s ease-out;
            position: relative;
            overflow: hidden;
        }
        
        .payment-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--warning), #667eea);
        }
        
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .payment-header {
            margin-bottom: 35px;
        }
        
        .payment-logo {
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, var(--warning), #ff6b35);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        .payment-logo i {
            font-size: 32px;
            color: white;
        }
        
        h2 {
            color: var(--text-dark);
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        
        .subtitle {
            color: var(--text-light);
            font-size: 16px;
            margin-bottom: 30px;
        }
        
        .payment-details {
            background: var(--bg-light);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            border: 2px solid var(--border);
        }
        
        .payment-info {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .info-item {
            text-align: left;
        }
        
        .info-label {
            font-size: 12px;
            color: var(--text-light);
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .info-value {
            font-size: 16px;
            color: var(--text-dark);
            font-weight: 600;
        }
        
        .amount-highlight {
            font-size: 24px;
            color: var(--warning);
            font-weight: 700;
        }
        
        .merchant-highlight {
            color: var(--primary);
            font-weight: 700;
        }
        
        .steps-section {
            margin: 30px 0;
        }
        
        .steps-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        
        .step-card {
            background: var(--bg-light);
            padding: 15px;
            border-radius: 12px;
            border: 2px solid var(--border);
            transition: all 0.3s ease;
        }
        
        .step-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            border-color: var(--primary);
        }
        
        .step-number {
            width: 30px;
            height: 30px;
            background: var(--warning);
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 14px;
            margin: 0 auto 10px;
        }
        
        .step-title {
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 6px;
            color: var(--text-dark);
        }
        
        .step-description {
            font-size: 11px;
            color: var(--text-medium);
            line-height: 1.4;
        }
        
        .camera-section {
            margin-bottom: 30px;
        }
        
        .camera-container {
            position: relative;
            width: 100%;
            max-width: 400px;
            margin: 0 auto;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        }
        
        #video {
            width: 100%;
            height: auto;
            display: block;
            transform: scaleX(-1);
        }
        
        .camera-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            pointer-events: none;
        }
        
        .face-outline {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 180px;
            height: 220px;
            border: 3px solid var(--warning);
            border-radius: 50%;
            opacity: 0.8;
            animation: breathe 2s infinite;
        }
        
        @keyframes breathe {
            0%, 100% { transform: translate(-50%, -50%) scale(1); }
            50% { transform: translate(-50%, -50%) scale(1.05); }
        }
        
        .face-outline.success {
            border: 3px solid var(--success);
            box-shadow: 0 0 0 3px rgba(56, 161, 105, 0.3);
        }
        
        .recording-indicator {
            position: absolute;
            top: 16px;
            left: 16px;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background-color: #e74c3c;
            display: none;
        }
        
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.3; }
            100% { opacity: 1; }
        }
        
        .recording-indicator.active {
            display: block;
            animation: blink 1s infinite;
        }
        
        .status-section {
            margin: 25px 0;
            min-height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .status-message {
            padding: 15px 20px;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 12px;
            max-width: 100%;
            text-align: center;
        }
        
        .status-info {
            background: var(--primary-light);
            color: var(--primary);
            border: 1px solid rgba(102, 126, 234, 0.3);
        }
        
        .status-success {
            background: #f0fff4;
            color: var(--success);
            border: 1px solid #c6f6d5;
        }
        
        .status-warning {
            background: #fffbeb;
            color: var(--warning);
            border: 1px solid #fbd38d;
        }
        
        .status-error {
            background: #fff5f5;
            color: var(--error);
            border: 1px solid #fed7d7;
        }
        
        .progress-container {
            margin: 20px 0;
            width: 100%;
            display: none;
        }
        
        .progress-bar {
            height: 12px;
            background-color: #ecf0f1;
            border-radius: 6px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(45deg, var(--warning), #ff6b35);
            width: 0%;
            transition: width 0.3s ease;
            border-radius: 6px;
        }
        
        .controls {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 25px;
        }
        
        .btn {
            padding: 15px 25px;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            text-decoration: none;
            min-width: 160px;
            justify-content: center;
        }
        
        .btn-primary {
            background: var(--primary);
            color: white;
        }
        
        .btn-primary:hover {
            background: var(--primary-dark);
            transform: translateY(-2px);
        }
        
        .btn-primary:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-warning {
            background: var(--warning);
            color: white;
        }
        
        .btn-warning:hover {
            background: #d68910;
            transform: translateY(-2px);
        }
        
        .btn-secondary {
            background: white;
            color: var(--text-medium);
            border: 2px solid var(--border);
        }
        
        .btn-secondary:hover {
            background: var(--bg-light);
            border-color: var(--primary);
        }
        
        .verify-btn {
            background-color: var(--warning);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 180px;
            gap: 10px;
        }
        
        .verify-btn:hover {
            background-color: #d68910;
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(237, 137, 54, 0.3);
        }
        
        .verify-btn:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .message {
            padding: 15px 18px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 500;
        }
        
        .success-message {
            background-color: #f0fff4;
            color: var(--success);
            border: 1px solid #c6f6d5;
        }
        
        .error-message {
            background-color: #fff5f5;
            color: var(--error);
            border: 1px solid #fed7d7;
        }
        
        .cancel-section {
            margin-top: 30px;
            padding-top: 25px;
            border-top: 1px solid var(--border);
            text-align: center;
        }
        
        .cancel-section p {
            color: var(--text-light);
            font-size: 14px;
            margin-bottom: 15px;
        }
        
        .cancel-btn {
            background: #95a5a6;
            color: white;
            border: 2px solid #95a5a6;
            padding: 12px 20px;
            border-radius: 10px;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .cancel-btn:hover {
            background: #7f8c8d;
            border-color: #7f8c8d;
            text-decoration: none;
            color: white;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }
            
            .payment-container {
                padding: 25px 20px;
            }
            
            .payment-info {
                grid-template-columns: 1fr;
                gap: 10px;
            }
            
            .camera-container {
                max-width: 100%;
            }
            
            .controls {
                flex-direction: column;
                align-items: center;
            }
            
            .btn {
                width: 100%;
                max-width: 250px;
            }
            
            .steps-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="payment-container">
            <div class="payment-header">
                <div class="payment-logo">
                    <i class="fas fa-credit-card"></i>
                </div>
                <h2>Payment Authorization</h2>
                <p class="subtitle">Verify your identity to complete this payment</p>
            </div>
            
            <div class="payment-details">
                <div class="payment-info">
                    <div class="info-item">
                        <div class="info-label">Merchant</div>
                        <div class="info-value merchant-highlight"><?= htmlspecialchars($payment_data['merchant_name']) ?></div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Order ID</div>
                        <div class="info-value"><?= htmlspecialchars($payment_data['order_id']) ?></div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Description</div>
                        <div class="info-value"><?= htmlspecialchars($payment_data['description']) ?></div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Amount</div>
                        <div class="info-value amount-highlight"><?= $payment_data['currency'] ?> <?= number_format($payment_data['amount'], 2) ?></div>
                    </div>
                </div>
            </div>
            
            <?php if ($error): ?>
                <div class="message error-message">
                    <i class="fas fa-exclamation-circle"></i>
                    <?= htmlspecialchars($error) ?>
                </div>
            <?php endif; ?>
            
            <?php if ($success): ?>
                <div class="message success-message">
                    <i class="fas fa-check-circle"></i>
                    <?= htmlspecialchars($success) ?>
                </div>
            <?php endif; ?>
            
            <div class="steps-section">
                <div class="steps-grid">
                    <div class="step-card">
                        <div class="step-number">1</div>
                        <div class="step-title">Face Verification</div>
                        <div class="step-description">Verify your identity with facial recognition</div>
                    </div>
                    <div class="step-card">
                        <div class="step-number">2</div>
                        <div class="step-title">Liveness Check</div>
                        <div class="step-description">Prove you're real with natural movement</div>
                    </div>
                    <div class="step-card">
                        <div class="step-number">3</div>
                        <div class="step-title">PIN Entry</div>
                        <div class="step-description">Enter your security PIN to authorize payment</div>
                    </div>
                </div>
            </div>
            
            <div class="camera-section" id="cameraSection">
                <div class="camera-container">
                    <video id="video" autoplay playsinline></video>
                    <div class="camera-overlay">
                        <div class="face-outline" id="faceOutline"></div>
                        <div class="recording-indicator" id="recordingIndicator"></div>
                    </div>
                </div>
            </div>
            
            <div class="status-section">
                <div class="status-message status-info" id="statusMessage">
                    Ready to start payment authorization
                </div>
            </div>
            
            <div class="progress-container" id="progressContainer">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
            </div>
            
            <div class="controls">
                <button class="btn btn-primary" id="startScanBtn">
                    <i class="fas fa-camera"></i>
                    Start Verification
                </button>
                
                <button class="verify-btn" id="verifyButton" style="display: none;">
                    <i class="fas fa-shield-alt"></i> Verify for Payment
                </button>
                
                <button class="btn btn-secondary" id="retryBtn" style="display: none;">
                    <i class="fas fa-redo"></i>
                    Try Again
                </button>
            </div>
            
            <form method="POST" id="verificationForm" style="display: none;">
                <input type="hidden" name="csrf_token" value="<?= $csrf_token ?>">
                <input type="hidden" name="action" value="verify_payment_face">
                <input type="hidden" name="user_id" id="hiddenUserId">
                <input type="hidden" name="similarity" id="hiddenSimilarity">
                <input type="hidden" name="liveness_score" id="hiddenLivenessScore">
                <input type="hidden" name="liveness_verified" id="hiddenLivenessVerified">
            </form>
            
            <div class="cancel-section">
                <p>Having trouble or want to cancel?</p>
                <button type="button" class="cancel-btn" onclick="cancelPayment()">
                    <i class="fas fa-times"></i>
                    Cancel Payment
                </button>
            </div>
        </div>
    </div>

    <script>
        // Configuration
        const API_BASE_URL = "http://localhost:5000";
        
        // ✅ SAFE PHP DATA TRANSFER
        const paymentData = <?= json_encode([
            'order_id' => $payment_data['order_id'] ?? '',
            'amount' => $payment_data['amount'] ?? 0,
            'currency' => $payment_data['currency'] ?? '',
            'merchant_name' => $payment_data['merchant_name'] ?? '',
            'merchant_main_url' => $payment_data['merchant_main_url'] ?? '',
            'cancel_url' => $payment_data['cancel_url'] ?? ''
        ]) ?>;
        
        const hasSuccess = <?= $success ? 'true' : 'false' ?>;
        
        // Global variables
        let stream;
        let frames = [];
        let faceDetected = false;
        let faceDetectionInterval;
        let retryCount = 0;
        const maxRetries = 3; // Increased retry limit since rate limiting is removed
        
        // Initialize page
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('startScanBtn').addEventListener('click', startFacialScan);
            document.getElementById('retryBtn').addEventListener('click', retryProcess);
            
            document.getElementById('verifyButton').addEventListener('click', function() {
                if (faceDetected) {
                    startLivenessChallenge();
                } else {
                    updateStatus("⚠️ No face detected. Please position your face in the outline.", "warning");
                }
            });
            
            // Initialize camera on load
            initializeCameraOnLoad();
        });
        
        // Initialize camera when page loads
        async function initializeCameraOnLoad() {
            try {
                await initializeWebcam();
                updateStatus('Camera ready! Click "Start Verification" to begin payment authorization.', 'info');
            } catch (error) {
                console.error('Error initializing camera on load:', error);
                updateStatus('Camera not available. Click "Start Verification" to try again.', 'warning');
            }
        }
        
        // Start facial scan for payment
        async function startFacialScan() {
            try {
                updateStatus('Starting payment verification...', 'info');
                
                if (!stream) {
                    await initializeWebcam();
                }
                
                startFaceDetection();
                document.getElementById('startScanBtn').style.display = 'none';
                updateStatus('Position your face in the outline to verify for payment', 'info');
                
            } catch (error) {
                console.error('Error starting facial scan:', error);
                updateStatus('Failed to access camera. Please check permissions.', 'error');
                showRetryOption();
            }
        }
        
        // Initialize webcam
        async function initializeWebcam() {
            const video = document.getElementById('video');
            
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 }, 
                    height: { ideal: 480 }, 
                    facingMode: 'user' 
                }
            });
            
            video.srcObject = stream;
            
            return new Promise((resolve) => {
                video.onloadedmetadata = () => {
                    video.play();
                    resolve();
                };
            });
        }
        
        // Start face detection
        function startFaceDetection() {
            fetch(`${API_BASE_URL}/health`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'healthy' || data.status === 'degraded') {
                        faceDetectionInterval = setInterval(checkFaceDetection, 1000);
                        updateStatus("Position your face in the outline for payment verification.", "info");
                    } else {
                        updateStatus("⚠️ Face detection API is not fully operational.", "error");
                        showRetryOption();
                    }
                })
                .catch(error => {
                    console.error("API health check failed:", error);
                    updateStatus("❌ Cannot connect to face detection API.", "error");
                    showRetryOption();
                });
        }
        
        // Check face detection
        function checkFaceDetection() {
            if (!document.getElementById('video').videoWidth) return;
            
            const canvas = document.createElement("canvas");
            canvas.width = document.getElementById('video').videoWidth;
            canvas.height = document.getElementById('video').videoHeight;
            const ctx = canvas.getContext("2d");
            ctx.scale(-1, 1);
            ctx.drawImage(document.getElementById('video'), -canvas.width, 0, canvas.width, canvas.height);
            
            const imageData = canvas.toDataURL("image/jpeg").split(',')[1];
            
            fetch(`${API_BASE_URL}/recognize`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image_base64: imageData })
            })
            .then(res => {
                if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
                return res.json();
            })
            .then(data => {
                if (data.status === "failed" && (
                    data.message.includes("No face detected") ||
                    data.message.includes("Face extraction failed") ||
                    data.message.includes("Detected face too small") ||
                    data.message.includes("Low face detection confidence"))
                ) {
                    if (faceDetected) {
                        faceDetected = false;
                        document.getElementById('faceOutline').classList.remove('success');
                        updateStatus('Position your face in the outline for payment verification', 'info');
                        hideVerifyButton();
                    }
                } else {
                    if (!faceDetected) {
                        faceDetected = true;
                        document.getElementById('faceOutline').classList.add('success');
                        updateStatus('Face detected! Click "Verify for Payment" when ready.', 'success');
                        showVerifyButton();
                    }
                }
            })
            .catch(error => {
                console.error("Face detection error:", error);
                faceDetected = false;
                document.getElementById('faceOutline').classList.remove('success');
                hideVerifyButton();
            });
        }
        
        // Show verify button
        function showVerifyButton() {
            const verifyButton = document.getElementById('verifyButton');
            verifyButton.style.display = 'inline-flex';
            verifyButton.disabled = !faceDetected;
        }
        
        // Hide verify button
        function hideVerifyButton() {
            const verifyButton = document.getElementById('verifyButton');
            verifyButton.style.display = 'none';
        }
        
        // Start liveness challenge for payment
        async function startLivenessChallenge() {
            try {
                clearInterval(faceDetectionInterval);
                hideVerifyButton();
                
                updateStatus('Ready for liveness challenge! Please move your face naturally.', 'info');
                
                setTimeout(() => {
                    captureFrames();
                }, 2000);
                
            } catch (error) {
                console.error('Error starting liveness challenge:', error);
                updateStatus('Failed to start liveness challenge. Please try again.', 'error');
                showRetryOption();
            }
        }
        
        // Capture frames for payment verification
        function captureFrames() {
            frames = [];
            let captured = 0;
            const total = 20;
            document.getElementById('progressContainer').style.display = 'block';
            document.getElementById('progressFill').style.width = '0%';
            updateStatus('📸 Capturing frames for payment... Please move your face naturally left and right.', 'info');
            document.getElementById('recordingIndicator').classList.add('active');
            
            const randomChallengeFrame = Math.floor(Math.random() * (total - 5)) + 3;
            let challengeShown = false;
            let challengeType = Math.random() > 0.5 ? 'blink' : 'look_up';
            
            const interval = setInterval(() => {
                const timestamp = new Date().getTime();
                const canvas = document.createElement("canvas");
                canvas.width = document.getElementById('video').videoWidth;
                canvas.height = document.getElementById('video').videoHeight;
                const ctx = canvas.getContext("2d");
                ctx.scale(-1, 1);
                ctx.drawImage(document.getElementById('video'), -canvas.width, 0, canvas.width, canvas.height);
                
                const frameData = {
                    image: canvas.toDataURL("image/jpeg"),
                    timestamp: timestamp,
                    challenge: challengeShown ? challengeType : null
                };
                
                frames.push(frameData);
                captured++;
                document.getElementById('progressFill').style.width = `${(captured / total) * 100}%`;
                
                if (captured === randomChallengeFrame && !challengeShown) {
                    challengeShown = true;
                    if (challengeType === 'blink') {
                        updateStatus("Now, please blink quickly for payment verification!", "warning");
                    } else {
                        updateStatus("Now, please look up briefly for payment verification!", "warning");
                    }
                }
                
                if (captured >= total) {
                    clearInterval(interval);
                    verifyPaymentWithAPI();
                }
            }, 250);
        }
        
        // Verify payment with API
        function verifyPaymentWithAPI() {
            updateStatus('🔍 Analyzing your identity for payment authorization...', 'info');
            document.getElementById('recordingIndicator').classList.remove('active');
            
            if (frames.length < 6) {
                retryCount++;
                updateStatus("❌ Not enough frames captured for payment. Please try again.", "error");
                handleRetryOrFail();
                return;
            }
            
            const mainImageData = frames[0].image.split(',')[1];
            const additionalFramesData = frames.slice(1).map(frame => frame.image.split(',')[1]);
            const timestamps = frames.map(frame => frame.timestamp);
            const timeDiffs = [];
            for (let i = 1; i < timestamps.length; i++) {
                timeDiffs.push(timestamps[i] - timestamps[i-1]);
            }
            const avgTimeDiff = timeDiffs.reduce((sum, diff) => sum + diff, 0) / timeDiffs.length;
            const timingVariance = Math.sqrt(
                timeDiffs.reduce((sum, diff) => sum + Math.pow(diff - avgTimeDiff, 2), 0) / timeDiffs.length
            );
            
            const authData = {
                image_base64: mainImageData,
                additional_frames: additionalFramesData,
                challenge_type: 'head_movement',
                timing_data: {
                    timestamps: timestamps,
                    avg_diff: avgTimeDiff,
                    variance: timingVariance
                },
                client_random: Math.random().toString().substring(2, 10)
            };
            
            const challengeFrames = frames.filter(frame => frame.challenge !== null);
            if (challengeFrames.length > 0) {
                authData.challenge_data = {
                    type: challengeFrames[0].challenge,
                    frame_index: frames.findIndex(frame => frame.challenge !== null)
                };
            }
            
            fetch(`${API_BASE_URL}/authenticate`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(authData)
            })
            .then(res => {
                if (!res.ok) {
                    throw new Error(`HTTP error! status: ${res.status}`);
                }
                return res.json();
            })
            .then(data => {
                if (data.status === "success" && data.liveness_verified && data.user_id) {
                    updateStatus("✅ Payment verification successful! Proceeding to PIN entry...", "success");
                    
                    document.getElementById('hiddenUserId').value = data.user_id;
                    document.getElementById('hiddenSimilarity').value = data.similarity || 0.9;
                    document.getElementById('hiddenLivenessScore').value = data.liveness_score || 0.8;
                    document.getElementById('hiddenLivenessVerified').value = 'true';
                    
                    setTimeout(() => {
                        document.getElementById('verificationForm').submit();
                    }, 1500);
                } else {
                    retryCount++;
                    if (!data.liveness_verified) {
                        if (data.message && data.message.includes("spoof")) {
                            updateStatus("🚫 Spoofing attempt detected. Payment verification failed.", "error");
                        } else if (data.message && data.message.includes("static")) {
                            updateStatus("🚫 Video replay detected. Live verification required for payment.", "error");
                        } else if (data.message && data.message.includes("Static image detected")) {
                            updateStatus("❌ Static image detected! Please use a live camera for payment.", "error");
                        } else if (data.message && data.message.includes("Video replay detected")) {
                            updateStatus("❌ Video replay detected! Please use a live camera for payment.", "error");
                        } else if (data.message && data.message.includes("Deepfake detected")) {
                            updateStatus("❌ Deepfake detected! Please use your real face for payment.", "error");
                        } else {
                            updateStatus(`❌ Payment security check failed: ${data.message || "Please follow the instructions carefully."}`, "error");
                        }
                    } else if (!data.user_id) {
                        updateStatus("❌ Face not recognized for payment. Please ensure you have registered your face first.", "error");
                    } else {
                        updateStatus(`❌ Payment verification failed: ${data.message || "Unknown error"}`, "error");
                    }
                    handleRetryOrFail();
                }
            })
            .catch(error => {
                console.error("Payment API error:", error);
                retryCount++;
                updateStatus("❌ Server error during payment verification. Please try again later.", "error");
                handleRetryOrFail();
            });
        }
        
        // Handle retry or fail for payment
        function handleRetryOrFail() {
            if (retryCount >= maxRetries) {
                updateStatus(`⚠️ Face verification failed after ${maxRetries} attempts. Redirecting to merchant...`, "error");
                setTimeout(() => {
                    const cancelUrl = paymentData.merchant_main_url || paymentData.cancel_url;
                    const orderId = paymentData.order_id;
                    window.location.href = cancelUrl + '?status=failed&reason=face_verification_failed&order_id=' + encodeURIComponent(orderId);
                }, 3000);
            } else {
                updateStatus(`❌ Face verification failed. ${maxRetries - retryCount} attempts remaining.`, "warning");
                showRetryOption();
            }
        }
        
        // Update status
        function updateStatus(message, type = 'info') {
            const statusEl = document.getElementById('statusMessage');
            statusEl.textContent = message;
            statusEl.className = `status-message status-${type}`;
        }
        
        // Show retry option
        function showRetryOption() {
            document.getElementById('retryBtn').style.display = 'inline-flex';
        }
        
        // Retry process
        function retryProcess() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            
            if (faceDetectionInterval) {
                clearInterval(faceDetectionInterval);
            }
            
            document.getElementById('progressContainer').style.display = 'none';
            document.getElementById('progressFill').style.width = '0%';
            
            document.getElementById('startScanBtn').style.display = 'inline-flex';
            document.getElementById('retryBtn').style.display = 'none';
            hideVerifyButton();
            
            frames = [];
            faceDetected = false;
            
            updateStatus(`Ready to retry face verification. ${maxRetries - retryCount} attempts remaining.`, 'info');
        }
        
        // ✅ Cancel payment function - CLEAN VERSION
        function cancelPayment() {
            if (confirm('Are you sure you want to cancel this payment?')) {
                // Clean up camera resources
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                }
                if (faceDetectionInterval) {
                    clearInterval(faceDetectionInterval);
                }
                
                // Clear client-side data
                if (typeof(Storage) !== "undefined") {
                    sessionStorage.removeItem('face_verified');
                    sessionStorage.removeItem('camera_initialized');
                }
                
                // Redirect to payment failed page
                const params = new URLSearchParams();
                params.append('status', 'cancelled');
                params.append('order_id', paymentData.order_id);
                params.append('amount', paymentData.amount);
                params.append('currency', paymentData.currency);
                params.append('merchant_name', paymentData.merchant_name);
                params.append('merchant_url', paymentData.merchant_main_url);
                params.append('error', 'Payment cancelled by user during face verification');
                
                window.location.href = 'payment_failed.php?' + params.toString();
            }
        }
        
        // Handle successful PHP response
        if (hasSuccess) {
            setTimeout(() => {
                window.location.href = 'payment_pin.php';
            }, 3000);
        }
        
        // Clean up on page unload
        window.addEventListener('beforeunload', () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            if (faceDetectionInterval) {
                clearInterval(faceDetectionInterval);
            }
        });
    </script>
</body>
</html>