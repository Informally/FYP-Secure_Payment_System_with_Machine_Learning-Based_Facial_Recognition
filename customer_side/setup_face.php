<?php
// setup_face.php - Face Registration/Re-registration System
require_once 'config.php';
require_once 'encrypt.php';

// Check authentication
if (!SessionManager::start() || !SessionManager::requireAuthLevel('basic')) {
    header("Location: login.php");
    exit();
}

$user_id = $_SESSION['authenticatedUserId'];
$user_db_id = $_SESSION['user_db_id'];
$username = $_SESSION['username'];

$success = "";
$error = "";
$mode = $_GET['mode'] ?? 'register'; // 'register' or 'update'

try {
    $conn = dbConnect(); // Assuming this returns mysqli connection
    
    // Get user details
    $stmt = $conn->prepare("SELECT full_name, email, account_status FROM users WHERE user_id = ?");
    $stmt->bind_param("s", $user_id);
    $stmt->execute();
    $result = $stmt->get_result();
    $user = $result->fetch_assoc();
    $stmt->close();
    
    // Check if face is already registered
    $stmt = $conn->prepare("SELECT id, created_at FROM face_embeddings WHERE user_id = ?");
    $stmt->bind_param("i", $user_db_id);
    $stmt->execute();
    $result = $stmt->get_result();
    $existing_face = $result->fetch_assoc();
    $stmt->close();
    
    $is_update = ($mode === 'update' && $existing_face);
    $page_title = $is_update ? 'Re-register Face' : 'Register Face';
    $page_description = $is_update ? 'Update your facial recognition data' : 'Set up facial recognition for secure login';
    
} catch (Exception $e) {
    error_log("Setup face error: " . $e->getMessage());
    $error = "Unable to load face registration data.";
}

// Handle face registration result
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $action = $_POST['action'] ?? '';
    
    if ($action === 'register_face') {
        // CSRF validation
        if (!validateCSRFToken($_POST['csrf_token'] ?? '')) {
            $error = "Invalid request. Please try again.";
        } else {
            $registration_successful = ($_POST['registration_successful'] ?? 'false') === 'true';
            $liveness_verified = ($_POST['liveness_verified'] ?? 'false') === 'true';
            $liveness_score = floatval($_POST['liveness_score'] ?? 0);
            
            if (!$registration_successful) {
                $error = "Face registration failed. Please try again.";
            } elseif (!$liveness_verified) {
                $error = "Liveness verification failed. Please ensure you're moving your face naturally.";
            } else {
                try {
                    if ($is_update) {
                        // Log the re-registration
                        SecurityUtils::logSecurityEvent($user_id, 'face_reregistration', 'success', [
                            'previous_registration' => $existing_face['created_at'],
                            'liveness_score' => $liveness_score
                        ]);
                        
                        $success = "Face re-registration successful! Your facial recognition data has been updated.";
                    } else {
                        // Log the initial registration
                        SecurityUtils::logSecurityEvent($user_id, 'face_registration', 'success', [
                            'liveness_score' => $liveness_score
                        ]);
                        
                        // NEW: Activate account after successful face setup (only for initial registration)
                        $stmt = $conn->prepare("UPDATE users SET account_status = 'active' WHERE id = ?");
                        $stmt->bind_param("i", $user_db_id);
                        $stmt->execute();
                        $stmt->close();
                        
                        error_log("Account activated for user_id: " . $user_id . " (db_id: " . $user_db_id . ")");
                        
                        $success = "Face registration successful! Your account is now fully activated and you can use facial recognition to log in.";
                    }
                    
                    // Refresh face data
                    $stmt = $conn->prepare("SELECT id, created_at FROM face_embeddings WHERE user_id = ?");
                    $stmt->bind_param("i", $user_db_id);
                    $stmt->execute();
                    $result = $stmt->get_result();
                    $existing_face = $result->fetch_assoc();
                    $stmt->close();
                    
                } catch (Exception $e) {
                    error_log("Face registration log error: " . $e->getMessage());
                    $error = "Registration completed but logging failed.";
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
    <title><?= $page_title ?> - Secure FacePay</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --primary: #3498db;
            --primary-light: #e3f2fd;
            --primary-dark: #2980b9;
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
            background-color: var(--bg-light);
            color: var(--text-dark);
            line-height: 1.6;
        }
        
        /* Header */
        .header {
            background: linear-gradient(135deg, var(--primary), #667eea);
            color: white;
            padding: 20px 0;
            box-shadow: 0 2px 10px var(--shadow);
        }
        
        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
            display: flex;
            align-items: center;
        }
        
        .back-btn {
            color: white;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 8px;
            transition: background-color 0.3s;
        }
        
        .back-btn:hover {
            background-color: rgba(255,255,255,0.1);
            text-decoration: none;
            color: white;
        }
        
        .page-title {
            font-size: 24px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
            margin-left: 300px;
        }
        
        /* Main Content */
        .main-content {
            max-width: 900px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        
        .setup-container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.1);
            text-align: center;
            animation: fadeUp 0.6s ease-out;
        }
        
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .setup-header {
            margin-bottom: 35px;
        }
        
        .setup-icon {
            width: 100px;
            height: 100px;
            background: linear-gradient(135deg, var(--primary), #667eea);
            border-radius: 25px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 25px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        .setup-icon i {
            font-size: 40px;
            color: white;
        }
        
        .setup-title {
            font-size: 32px;
            font-weight: 700;
            color: var(--text-dark);
            margin-bottom: 12px;
        }
        
        .setup-description {
            font-size: 18px;
            color: var(--text-light);
            line-height: 1.6;
            max-width: 600px;
            margin: 0 auto;
        }
        
        /* Account Status Info */
        .account-status-info {
            background: var(--primary-light);
            color: var(--primary);
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            gap: 12px;
            text-align: left;
            border: 1px solid rgba(52, 152, 219, 0.3);
        }
        
        .account-status-info.pending {
            background: #fffbeb;
            color: var(--warning);
            border: 1px solid #fbd38d;
        }
        
        .account-status-info i {
            font-size: 20px;
            flex-shrink: 0;
        }
        
        .account-status-details h4 {
            margin-bottom: 4px;
            font-size: 14px;
            font-weight: 600;
        }
        
        .account-status-details p {
            font-size: 13px;
            opacity: 0.9;
            line-height: 1.4;
        }
        
        /* Existing Face Info */
        .existing-face-info {
            background: var(--warning);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            gap: 15px;
            text-align: left;
        }
        
        .existing-face-info i {
            font-size: 24px;
            flex-shrink: 0;
        }
        
        .existing-face-details h4 {
            margin-bottom: 5px;
            font-size: 16px;
        }
        
        .existing-face-details p {
            font-size: 14px;
            opacity: 0.9;
        }
        
        /* Steps Section */
        .steps-section {
            margin: 40px 0;
        }
        
        .steps-title {
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 25px;
            color: var(--text-dark);
        }
        
        .steps-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }
        
        .step-card {
            background: var(--bg-light);
            padding: 25px;
            border-radius: 15px;
            border: 2px solid var(--border);
            transition: all 0.3s ease;
        }
        
        .step-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            border-color: var(--primary);
        }
        
        .step-number {
            width: 40px;
            height: 40px;
            background: var(--primary);
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 18px;
            margin: 0 auto 15px;
        }
        
        .step-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 10px;
            color: var(--text-dark);
        }
        
        .step-description {
            font-size: 14px;
            color: var(--text-medium);
            line-height: 1.5;
        }
        
        /* Camera Section - Always visible */
        .camera-section {
            margin: 40px 0;
        }
        
        .camera-container {
            position: relative;
            width: 100%;
            max-width: 480px;
            margin: 0 auto;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        }
        
        #video {
            width: 100%;
            height: auto;
            display: block;
            transform: scaleX(-1); /* Mirror effect */
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
            width: 200px;
            height: 240px;
            border: 3px solid var(--primary);
            border-radius: 50%;
            opacity: 0.8;
            animation: breathe 2s infinite;
        }
        
        @keyframes breathe {
            0%, 100% { transform: translate(-50%, -50%) scale(1); }
            50% { transform: translate(-50%, -50%) scale(1.05); }
        }
        
        .face-outline.success {
            border: 3px solid #27ae60;
            box-shadow: 0 0 0 3px rgba(39, 174, 96, 0.3);
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
        
        /* Status Messages */
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
            border: 1px solid rgba(52, 152, 219, 0.3);
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
        
        /* Progress Bar */
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
            background: linear-gradient(45deg, #27ae60, #2ecc71);
            width: 0%;
            transition: width 0.3s ease;
            border-radius: 6px;
        }
        
        /* Controls */
        .controls {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 30px;
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
        
        .btn-secondary {
            background: white;
            color: var(--text-medium);
            border: 2px solid var(--border);
        }
        
        .btn-secondary:hover {
            background: var(--bg-light);
            border-color: var(--primary);
        }
        
        .btn-success {
            background: var(--success);
            color: white;
        }
        
        .btn-success:hover {
            background: #2f855a;
            transform: translateY(-2px);
        }
        
        .verify-btn {
            background-color: #4caf50;
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
            background-color: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(76, 175, 80, 0.3);
        }
        
        .verify-btn i {
            font-size: 16px;
        }
        
        .verify-btn:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        /* Messages */
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
        
        /* Security Info */
        .security-info {
            background: var(--primary-light);
            padding: 20px;
            border-radius: 12px;
            margin-top: 30px;
            text-align: left;
            border-left: 4px solid var(--primary);
        }
        
        .security-info h4 {
            color: var(--primary);
            margin-bottom: 10px;
            font-size: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .security-info ul {
            color: var(--text-medium);
            font-size: 14px;
            padding-left: 20px;
        }
        
        .security-info li {
            margin-bottom: 5px;
        }
        
        /* Spinner */
        .spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid transparent;
            border-top: 2px solid currentColor;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .main-content {
                padding: 20px 15px;
            }
            
            .setup-container {
                padding: 25px 20px;
            }
            
            .setup-title {
                font-size: 26px;
            }
            
            .setup-description {
                font-size: 16px;
            }
            
            .steps-grid {
                grid-template-columns: 1fr;
            }
            
            .controls {
                flex-direction: column;
                align-items: center;
            }
            
            .btn {
                width: 100%;
                max-width: 280px;
            }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="header-content">
            <a href="<?= $is_update ? 'profile.php' : 'dashboard.php' ?>" class="back-btn">
                <i class="fas fa-arrow-left"></i>
                Back to <?= $is_update ? 'Profile' : 'Dashboard' ?>
            </a>
            <div class="page-title">
                <i class="fas fa-user-check"></i>
                <?= $page_title ?>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="main-content">
        <div class="setup-container">
            <!-- Header -->
            <div class="setup-header">
                <div class="setup-icon">
                    <i class="fas fa-face-smile"></i>
                </div>
                <h1 class="setup-title"><?= $page_title ?></h1>
                <p class="setup-description">
                    <?= $page_description ?>. Simply move your face naturally in different directions while looking at the camera.
                </p>
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
            
            <!-- Account Status Info -->
            <?php if (!$is_update && isset($user['account_status'])): ?>
                <div class="account-status-info <?= $user['account_status'] === 'pending' ? 'pending' : '' ?>">
                    <i class="fas fa-<?= $user['account_status'] === 'pending' ? 'clock' : 'check-circle' ?>"></i>
                    <div class="account-status-details">
                        <h4>Account Status: <?= ucfirst($user['account_status']) ?></h4>
                        <p>
                            <?php if ($user['account_status'] === 'pending'): ?>
                                Your account will be automatically activated after completing face registration.
                            <?php else: ?>
                                Your account is active and ready to use.
                            <?php endif; ?>
                        </p>
                    </div>
                </div>
            <?php endif; ?>
            
            <?php if ($is_update && $existing_face): ?>
                <div class="existing-face-info">
                    <i class="fas fa-info-circle"></i>
                    <div class="existing-face-details">
                        <h4>Face Already Registered</h4>
                        <p>Your face was registered on <?= date('M j, Y', strtotime($existing_face['created_at'])) ?>. Re-registering will replace your current facial data.</p>
                    </div>
                </div>
            <?php endif; ?>
            
            <!-- Process Steps -->
            <div class="steps-section">
                <h3 class="steps-title">Registration Process</h3>
                <div class="steps-grid">
                    <div class="step-card">
                        <div class="step-number">1</div>
                        <div class="step-title">Face Detection</div>
                        <div class="step-description">Position your face in the camera frame</div>
                    </div>
                    <div class="step-card">
                        <div class="step-number">2</div>
                        <div class="step-title">Liveness Verification</div>
                        <div class="step-description">Move your face naturally to verify you're real</div>
                    </div>
                    <!-- <div class="step-card">
                        <div class="step-number">3</div>
                        <div class="step-title">Face Registration</div>
                        <div class="step-description">System saves your facial features securely</div>
                    </div> -->
                </div>
            </div>
            
            <!-- Camera Section - Always visible -->
            <div class="camera-section" id="cameraSection">
                <div class="camera-container">
                    <video id="video" autoplay playsinline></video>
                    <div class="camera-overlay">
                        <div class="face-outline" id="faceOutline"></div>
                        <div class="recording-indicator" id="recordingIndicator"></div>
                    </div>
                </div>
            </div>
            
            <!-- Status Messages -->
            <div class="status-section">
                <div class="status-message status-info" id="statusMessage">
                    Ready to start face registration
                </div>
            </div>
            
            <!-- Progress Bar -->
            <div class="progress-container" id="progressContainer">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
            </div>
            
            <!-- Controls -->
            <div class="controls">
                <button class="btn btn-primary" id="startScanBtn">
                    <i class="fas fa-camera"></i>
                    <?= $is_update ? 'Re-register Face' : 'Start Registration' ?>
                </button>
                
                <!-- Verify Button - Always present but initially hidden -->
                <button class="verify-btn" id="verifyButton" style="display: none;">
                    <i class="fas fa-camera"></i> Verify & Register
                </button>
                
                <button class="btn btn-secondary" id="retryBtn" style="display: none;">
                    <i class="fas fa-redo"></i>
                    Try Again
                </button>
                <a href="<?= $is_update ? 'profile.php' : 'dashboard.php' ?>" class="btn btn-secondary">
                    <i class="fas fa-times"></i>
                    Cancel
                </a>
            </div>
            
            <!-- Hidden form for submission -->
            <form method="POST" id="registrationForm" style="display: none;">
                <input type="hidden" name="csrf_token" value="<?= $csrf_token ?>">
                <input type="hidden" name="action" value="register_face">
                <input type="hidden" name="registration_successful" id="hiddenRegistrationSuccessful">
                <input type="hidden" name="liveness_verified" id="hiddenLivenessVerified">
                <input type="hidden" name="liveness_score" id="hiddenLivenessScore">
            </form>
            
            <!-- Security Information -->
            <div class="security-info">
                <h4><i class="fas fa-shield-alt"></i> Security & Privacy</h4>
                <ul>
                    <li>Advanced AI protection runs automatically in the background</li>
                    <li>System detects and prevents photo, video, and deepfake attacks</li>
                    <li>Your facial data is encrypted and stored securely</li>
                    <li>No actual photos are saved, only mathematical representations</li>
                    <li>You can re-register or delete your facial data anytime</li>
                    <?php if (!$is_update): ?>
                    <li><strong>Your account will be activated automatically after successful registration</strong></li>
                    <?php endif; ?>
                </ul>
            </div>
        </div>
    </main>

    <script>
        // Configuration - Same as kiosk
        const API_BASE_URL = "http://localhost:5000";
        const IS_UPDATE_MODE = <?= $is_update ? 'true' : 'false' ?>;
        
        // 🔥 FIX: Use consistent user_id from PHP session
        const PHP_USER_ID = '<?= $user_id ?>';
        console.log('Using PHP User ID:', PHP_USER_ID);
        
        // Global variables - Same as kiosk
        let stream;
        let frames = [];
        let faceDetected = false;
        let faceDetectionInterval;
        let retryCount = 0;
        const maxRetries = 3;
        
        // 🔥 Store the successful frame data for registration
        let successfulFrameData = null;
        
        // Initialize page and camera immediately
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('startScanBtn').addEventListener('click', startFacialScan);
            document.getElementById('retryBtn').addEventListener('click', retryProcess);
            
            // Add event listener for verify button
            document.getElementById('verifyButton').addEventListener('click', function() {
                if (faceDetected) {
                    startLivenessChallenge();
                } else {
                    updateStatus("⚠️ No face detected. Please position your face in the outline.", "warning");
                }
            });
            
            // Initialize camera immediately when page loads
            initializeCameraOnLoad();
        });
        
        // Initialize camera when page loads (without starting scan)
        async function initializeCameraOnLoad() {
            try {
                await initializeWebcam();
                updateStatus('Camera ready! Click "Start Registration" to begin.', 'info');
            } catch (error) {
                console.error('Error initializing camera on load:', error);
                updateStatus('Camera not available. Click "Start Registration" to try again.', 'warning');
            }
        }
        
        // Start facial scan - Now just starts detection since camera is already on
        async function startFacialScan() {
            try {
                updateStatus('Starting facial scan...', 'info');
                
                // If camera isn't already initialized, initialize it
                if (!stream) {
                    await initializeWebcam();
                }
                
                // Start face detection
                startFaceDetection();
                
                // Update UI - hide start button
                document.getElementById('startScanBtn').style.display = 'none';
                
                updateStatus('Position your face in the outline', 'info');
                
            } catch (error) {
                console.error('Error starting facial scan:', error);
                updateStatus('Failed to access camera. Please check permissions.', 'error');
                showRetryOption();
            }
        }
        
        // Initialize webcam - Same as kiosk
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
        
        // Start face detection - Same as kiosk
        function startFaceDetection() {
            // Check API availability first
            fetch(`${API_BASE_URL}/health`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'healthy' || data.status === 'degraded') {
                        faceDetectionInterval = setInterval(checkFaceDetection, 1000);
                        updateStatus("Position your face in the outline.", "info");
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
        
        // Check face detection - Same as kiosk
        function checkFaceDetection() {
            if (!document.getElementById('video').videoWidth) return;
            
            const canvas = document.createElement("canvas");
            canvas.width = document.getElementById('video').videoWidth;
            canvas.height = document.getElementById('video').videoHeight;
            const ctx = canvas.getContext("2d");
            ctx.scale(-1, 1); // Mirror like kiosk
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
                        updateStatus('Position your face in the outline', 'info');
                        
                        // Hide verify button
                        hideVerifyButton();
                    }
                } else {
                    if (!faceDetected) {
                        faceDetected = true;
                        document.getElementById('faceOutline').classList.add('success');
                        updateStatus('Face detected! Click "Verify & Register" when ready.', 'success');
                        
                        // Show verify button
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
        
        // Show verify button - Simplified
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
        
        // Start liveness challenge - Same as kiosk
        async function startLivenessChallenge() {
            try {
                clearInterval(faceDetectionInterval);
                
                // Hide verify button
                hideVerifyButton();
                
                updateStatus('Ready for liveness challenge! Please move your face naturally.', 'info');
                
                // Start capturing frames automatically after a short delay
                setTimeout(() => {
                    captureFrames();
                }, 2000);
                
            } catch (error) {
                console.error('Error starting liveness challenge:', error);
                updateStatus('Failed to start liveness challenge. Please try again.', 'error');
                showRetryOption();
            }
        }
        
        // Capture frames - Same method as kiosk
        function captureFrames() {
            frames = [];
            let captured = 0;
            const total = 20;
            document.getElementById('progressContainer').style.display = 'block';
            document.getElementById('progressFill').style.width = '0%';
            updateStatus('📸 Capturing frames... Please move your face naturally left and right.', 'info');
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
                ctx.scale(-1, 1); // Mirror like kiosk
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
                        updateStatus("Now, please blink quickly!", "warning");
                    } else {
                        updateStatus("Now, please look up briefly!", "warning");
                    }
                }
                
                if (captured >= total) {
                    clearInterval(interval);
                    verifyWithAPI();
                }
            }, 250);
        }
        
        // 🔥 FIXED: Verify with API and store successful frame
        function verifyWithAPI() {
            updateStatus('🔍 Analyzing your identity and liveness...', 'info');
            document.getElementById('recordingIndicator').classList.remove('active');
            
            if (frames.length < 6) {
                retryCount++;
                updateStatus("❌ Not enough frames captured. Please try again.", "error");
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
                security_only: true,  // Only run security verification, not face recognition
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
            
            // 🔥 Add logging for debugging
            console.log('Authentication data:', {
                user_id: PHP_USER_ID,
                security_only: authData.security_only,
                image_size: mainImageData.length,
                additional_frames: additionalFramesData.length
            });
            
            // Call /authenticate endpoint for security verification
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
                if (data.status === "success" && data.liveness_verified && data.security_only) {
                    // 🔥 FIX: Store the successful frame for registration
                    successfulFrameData = mainImageData;
                    
                    updateStatus("✅ Security verified! Registering your face...", "success");
                    setTimeout(() => {
                        registerFaceWithAPI(data.liveness_score || 0.8);
                    }, 1500);
                } else {
                    retryCount++;
                    // Handle specific error messages like kiosk
                    if (!data.liveness_verified) {
                        if (data.message && data.message.includes("spoof")) {
                            updateStatus("🚫 Spoofing attempt detected. Please try again with a real person.", "error");
                        } else if (data.message && data.message.includes("static")) {
                            updateStatus("🚫 Video replay detected. Live verification required.", "error");
                        } else if (data.message && data.message.includes("Static image detected")) {
                            updateStatus("❌ Static image detected! Please use a live camera.", "error");
                        } else if (data.message && data.message.includes("Video replay detected")) {
                            updateStatus("❌ Video replay detected! Please use a live camera.", "error");
                        } else if (data.message && data.message.includes("Deepfake detected")) {
                            updateStatus("❌ Deepfake detected! Please use your real face.", "error");
                        } else {
                            updateStatus(`❌ Security check failed: ${data.message || "Please follow the instructions carefully."}`, "error");
                        }
                    } else {
                        updateStatus(`❌ Verification failed: ${data.message || "Unknown error"}`, "error");
                    }
                    handleRetryOrFail();
                }
            })
            .catch(error => {
                console.error("API error:", error);
                retryCount++;
                updateStatus("❌ Server error during verification. Please try again later.", "error");
                handleRetryOrFail();
            });
        }
        
        // 🔥 FIXED: Register face using the SAME successful frame
        function registerFaceWithAPI(livenessScore) {
            try {
                updateStatus('📝 Registering your face...', 'info');
                
                // 🔥 KEY FIX: Use the stored successful frame, NOT a new capture
                if (!successfulFrameData) {
                    throw new Error('No successful frame data available for registration');
                }
                
                const registrationData = {
                    user_id: PHP_USER_ID, // 🔥 Use consistent user_id
                    image_base64: successfulFrameData, // 🔥 Use the SAME frame that passed authentication
                    override_existing: IS_UPDATE_MODE
                };
                
                // 🔥 Add logging for debugging
                console.log('Registration data:', {
                    user_id: registrationData.user_id,
                    override_existing: registrationData.override_existing,
                    image_size: successfulFrameData.length,
                    same_frame_as_auth: true
                });
                
                fetch(`${API_BASE_URL}/register`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(registrationData)
                })
                .then(res => {
                    if (!res.ok) {
                        throw new Error(`HTTP error! status: ${res.status}`);
                    }
                    return res.json();
                })
                .then(regData => {
                    if (regData.status === "success") {
                        updateStatus("✅ Face registration successful! Your face is now registered for secure payments.", "success");
                        
                        // Submit to PHP with success data
                        document.getElementById('hiddenRegistrationSuccessful').value = 'true';
                        document.getElementById('hiddenLivenessVerified').value = 'true';
                        document.getElementById('hiddenLivenessScore').value = livenessScore;
                        
                        setTimeout(() => {
                            document.getElementById('registrationForm').submit();
                        }, 1500);
                        
                    } else {
                        throw new Error(regData.message || 'Registration failed');
                    }
                    
                })
                .catch(error => {
                    console.error('Face registration error:', error);
                    updateStatus('❌ Registration failed: ' + error.message, 'error');
                    showRetryOption();
                });
                
            } catch (error) {
                console.error('Face registration setup error:', error);
                updateStatus('❌ Registration setup failed: ' + error.message, 'error');
                showRetryOption();
            }
        }
        
        // Handle retry or fail - Same as kiosk
        function handleRetryOrFail() {
            if (retryCount >= maxRetries) {
                updateStatus("⚠️ Too many failed attempts. Please try again later.", "error");
                setTimeout(() => {
                    window.location.href = '<?= $is_update ? "profile.php" : "dashboard.php" ?>';
                }, 3000);
            } else {
                showRetryOption();
            }
        }
        
        // Update status - Same as kiosk
        function updateStatus(message, type = 'info') {
            const statusEl = document.getElementById('statusMessage');
            statusEl.textContent = message;
            statusEl.className = `status-message status-${type}`;
        }
        
        // Show retry option - Same as kiosk
        function showRetryOption() {
            document.getElementById('retryBtn').style.display = 'inline-flex';
        }
        
        // Retry process - Same as kiosk
        function retryProcess() {
            // Reset everything
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            
            if (faceDetectionInterval) {
                clearInterval(faceDetectionInterval);
            }
            
            // Reset UI - keep camera visible
            document.getElementById('progressContainer').style.display = 'none';
            document.getElementById('progressFill').style.width = '0%';
            
            // Reset buttons
            document.getElementById('startScanBtn').style.display = 'inline-flex';
            document.getElementById('retryBtn').style.display = 'none';
            hideVerifyButton();
            
            // Reset variables
            frames = [];
            faceDetected = false;
            retryCount = 0;
            successfulFrameData = null; // 🔥 Reset successful frame data
            
            updateStatus('Ready to try again. Click "Start Registration" to begin.', 'info');
        }
        
        // Handle successful PHP response
        <?php if ($success): ?>
        setTimeout(() => {
            window.location.href = '<?= $is_update ? "profile.php" : "dashboard.php" ?>';
        }, 3000);
        <?php endif; ?>
        
        // Clean up on page unload - Same as kiosk
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