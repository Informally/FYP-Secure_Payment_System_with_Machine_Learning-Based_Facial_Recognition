<?php
// business_integration/shopping_cart.php
const MERCHANT_ID = 'MCD';
const FACEPAY_API_KEY = 'sk_1f55be80af366a8c3e87f53b14962a4f2ec4bbe14755af39';
const FACEPAY_GATEWAY_URL = 'http://localhost/FYP/gateway/checkout.php';
// ✅ NEW: No more separate success/cancel URLs - gateway handles everything!
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Checkout - McDonald's Kiosk</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Arial', sans-serif;
            background: #f5f5f5;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .kiosk-header {
            background: linear-gradient(45deg, #da291c, #ffc72c);
            padding: 20px 0;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }

        .kiosk-header h1 {
            color: white;
            font-size: 2.8rem;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .checkout-container {
            max-width: 1400px;
            margin: 30px auto;
            padding: 0 30px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            flex: 1;
        }

        .section-card {
            background: white;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border: 3px solid #ffc72c;
            height: fit-content;
        }

        .section-header {
            background: linear-gradient(45deg, #da291c, #ffc72c);
            color: white;
            padding: 25px;
            font-size: 26px;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .section-content {
            padding: 30px;
        }

        /* Order Summary Styles */
        .order-item {
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 20px 0;
            border-bottom: 2px solid #f0f0f0;
        }

        .item-image {
            width: 70px;
            height: 70px;
            background: linear-gradient(135deg, #ffc72c, #ffeb3b);
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            color: #da291c;
        }

        .item-details {
            flex: 1;
        }

        .item-name {
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
            font-size: 18px;
        }

        .item-price {
            color: #da291c;
            font-weight: bold;
            font-size: 18px;
        }

        .item-quantity {
            background: #da291c;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
        }

        .total-section {
            background: linear-gradient(135deg, #34495e, #2c3e50);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin-top: 25px;
        }

        .total-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 12px;
            font-size: 18px;
        }

        .total-row.final {
            font-size: 28px;
            font-weight: bold;
            border-top: 3px solid rgba(255,255,255,0.3);
            padding-top: 20px;
            margin-top: 20px;
        }

        /* Payment Options Grid */
        .payment-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 25px;
        }

        .payment-option {
            border: 3px solid #e0e0e0;
            border-radius: 15px;
            padding: 20px 15px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: white;
            position: relative;
            min-height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .payment-option.active {
            border-color: #27ae60;
            background: linear-gradient(135deg, #27ae60, #2ecc71);
            color: white;
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(39, 174, 96, 0.3);
        }

        .payment-option.disabled {
            background: #f8f9fa;
            border-color: #dee2e6;
            color: #6c757d;
            cursor: not-allowed;
            opacity: 0.6;
        }

        .payment-option.disabled::after {
            content: 'Not Available';
            position: absolute;
            top: 8px;
            right: 8px;
            background: #dc3545;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: bold;
        }

        .payment-icon {
            font-size: 36px;
            margin-bottom: 12px;
            display: block;
        }

        .payment-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 6px;
        }

        .payment-description {
            font-size: 12px;
            line-height: 1.3;
            opacity: 0.9;
        }

        /* Credit Card specific styling */
        .payment-option.credit-card .payment-icon {
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .payment-option.touch-go .payment-icon {
            color: #00b4d8;
        }

        .payment-option.grabpay .payment-icon {
            color: #00b14f;
        }

        .payment-option.boost .payment-icon {
            color: #ff6b35;
        }

        .payment-option.facepay .payment-icon {
            color: #27ae60;
        }

        /* Action Buttons */
        .action-buttons {
            display: flex;
            gap: 15px;
            margin-top: 25px;
        }

        .btn {
            flex: 1;
            padding: 15px 20px;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .btn-primary {
            background: linear-gradient(45deg, #27ae60, #2ecc71);
            color: white;
        }

        .btn-primary:hover:not(:disabled) {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(39, 174, 96, 0.4);
        }

        .btn-primary:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .btn-secondary {
            background: #95a5a6;
            color: white;
        }

        .btn-secondary:hover {
            background: #7f8c8d;
            transform: translateY(-2px);
        }

        /* Selected Payment Method Display */
        .selected-method {
            background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
            border: 3px solid #27ae60;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 20px;
            display: none;
        }

        .selected-method.show {
            display: block;
            animation: slideDown 0.3s ease-out;
        }

        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .selected-method h4 {
            color: #27ae60;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 16px;
        }

        .selected-method p {
            color: #2d5a3d;
            margin: 0;
            font-size: 14px;
        }

        /* ✅ NEW: Gateway Integration Notice */
        .gateway-notice {
            background: linear-gradient(135deg, #e3f2fd, #f3e5f5);
            border: 2px solid #2196f3;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
            text-align: center;
        }

        .gateway-notice h4 {
            color: #1565c0;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .gateway-notice p {
            color: #424242;
            font-size: 14px;
            line-height: 1.5;
        }

        /* Responsive Design */
        @media (max-width: 1024px) {
            .checkout-container {
                grid-template-columns: 1fr;
                gap: 30px;
                max-width: 800px;
            }
            
            .payment-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (max-width: 768px) {
            .kiosk-header h1 {
                font-size: 2.2rem;
            }
            
            .section-content {
                padding: 20px;
            }
            
            .payment-option {
                padding: 18px 12px;
                min-height: 120px;
            }
            
            .payment-grid {
                grid-template-columns: 1fr;
            }
            
            .btn {
                font-size: 14px;
                padding: 12px 16px;
            }
        }
    </style>
</head>
<body>
    <div class="kiosk-header">
        <h1><i class="fas fa-store"></i> McDonald's Kiosk</h1>
    </div>

    <div class="checkout-container">
        <!-- Order Summary -->
        <div class="section-card">
            <div class="section-header">
                <i class="fas fa-receipt"></i>
                Your Order
            </div>
            <div class="section-content">
                <div id="orderItems"></div>
                <div class="total-section">
                    <div class="total-row">
                        <span>Subtotal:</span>
                        <span>RM <span id="subtotal">0.00</span></span>
                    </div>
                    <div class="total-row">
                        <span>Tax (6% SST):</span>
                        <span>RM <span id="tax">0.00</span></span>
                    </div>
                    <div class="total-row final">
                        <span>Total:</span>
                        <span>RM <span id="finalTotal">0.00</span></span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Payment Options -->
        <div class="section-card">
            <div class="section-header">
                <i class="fas fa-credit-card"></i>
                Choose Payment Method
            </div>
            <div class="section-content">
                <!-- ✅ NEW: Gateway Integration Notice
                <div class="gateway-notice">
                    <h4><i class="fas fa-shield-alt"></i> Secure Payment Gateway</h4>
                    <p>All payments are processed through FacePay's secure gateway. After payment, you'll be returned here automatically.</p>
                </div> -->

                <!-- Selected Payment Method Display -->
                <div class="selected-method" id="selectedMethod">
                    <h4><i class="fas fa-check-circle"></i> Selected Payment Method</h4>
                    <p id="selectedMethodText">None selected</p>
                </div>

                <!-- Payment Options Grid -->
                <div class="payment-grid">
                    <!-- Credit/Debit Card -->
                    <div class="payment-option disabled credit-card" onclick="showNotAvailable('Credit/Debit Card')">
                        <i class="payment-icon fas fa-credit-card"></i>
                        <div class="payment-title">Credit/Debit Card</div>
                        <div class="payment-description">Visa, Mastercard, Amex</div>
                    </div>

                    <!-- Touch 'n Go -->
                    <div class="payment-option disabled touch-go" onclick="showNotAvailable('Touch n Go')">
                        <i class="payment-icon fas fa-mobile-alt"></i>
                        <div class="payment-title">Touch 'n Go</div>
                        <div class="payment-description">eWallet & Physical Card</div>
                    </div>

                    <!-- GrabPay -->
                    <div class="payment-option disabled grabpay" onclick="showNotAvailable('GrabPay')">
                        <i class="payment-icon fas fa-car"></i>
                        <div class="payment-title">GrabPay</div>
                        <div class="payment-description">Digital wallet by Grab</div>
                    </div>

                    <!-- Boost -->
                    <div class="payment-option disabled boost" onclick="showNotAvailable('Boost')">
                        <i class="payment-icon fas fa-rocket"></i>
                        <div class="payment-title">Boost</div>
                        <div class="payment-description">Mobile payment app</div>
                    </div>

                    <!-- FacePay Gateway - ONLY WORKING OPTION -->
                    <div class="payment-option facepay" id="facepayOption" onclick="selectFacePay()">
                        <i class="payment-icon fas fa-user-check"></i>
                        <div class="payment-title">FacePay</div>
                        <div class="payment-description">Secure facial recognition payment</div>
                    </div>

                    <!-- Cash (Disabled for kiosk demo) -->
                    <div class="payment-option disabled" onclick="showNotAvailable('Cash')">
                        <i class="payment-icon fas fa-money-bill-wave"></i>
                        <div class="payment-title">Cash</div>
                        <div class="payment-description">Pay with cash at counter</div>
                    </div>
                </div>

                <!-- ✅ UPDATED: Hidden gateway form with new architecture -->
                <form id="facepayGatewayForm" method="POST" action="<?= FACEPAY_GATEWAY_URL ?>" style="display: none;">
                    <input type="hidden" name="merchant_id" value="<?= MERCHANT_ID ?>">
                    <input type="hidden" name="api_key" value="<?= FACEPAY_API_KEY ?>">
                    <input type="hidden" name="amount" id="gateway_amount">
                    <input type="hidden" name="order_id" id="gateway_order_id">
                    <input type="hidden" name="currency" value="MYR">
                    <input type="hidden" name="description" id="gateway_description">
                </form>

                <div class="action-buttons">
                    <button class="btn btn-secondary" onclick="goBack()">
                        <i class="fas fa-arrow-left"></i> Back to Menu
                    </button>
                    <button class="btn btn-primary" id="proceedBtn" disabled onclick="proceedWithPayment()">
                        <i class="fas fa-lock"></i> Proceed to Payment
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let cart = [];
        let selectedPaymentMethod = null;

        document.addEventListener('DOMContentLoaded', function() {
            loadCartFromStorage();
            renderOrderSummary();
        });

        function loadCartFromStorage() {
            const savedCart = sessionStorage.getItem('kioskCart');
            if (savedCart) {
                cart = JSON.parse(savedCart);
            }
            if (cart.length === 0) {
                // Add demo items for presentation
                cart = [
                    {id: 1, name: "Big Mac", price: 15.90, quantity: 1, icon: "🍔"},
                    {id: 2, name: "Large Fries", price: 6.50, quantity: 1, icon: "🍟"},
                    {id: 3, name: "Coca-Cola", price: 4.90, quantity: 1, icon: "🥤"}
                ];
            }
        }

        function renderOrderSummary() {
            const orderItems = document.getElementById('orderItems');
            const subtotalEl = document.getElementById('subtotal');
            const taxEl = document.getElementById('tax'); 
            const finalTotalEl = document.getElementById('finalTotal');

            orderItems.innerHTML = cart.map(item => `
                <div class="order-item">
                    <div class="item-image">${item.icon}</div>
                    <div class="item-details">
                        <div class="item-name">${item.name}</div>
                        <div class="item-price">RM ${item.price.toFixed(2)}</div>
                    </div>
                    <div class="item-quantity">Qty: ${item.quantity}</div>
                </div>
            `).join('');

            const subtotal = cart.reduce((sum, item) => sum + (item.price * item.quantity), 0);
            const tax = subtotal * 0.06;
            const finalTotal = subtotal + tax;

            subtotalEl.textContent = subtotal.toFixed(2);
            taxEl.textContent = tax.toFixed(2);
            finalTotalEl.textContent = finalTotal.toFixed(2);
        }

        function selectFacePay() {
            // Remove active class from all options
            document.querySelectorAll('.payment-option').forEach(option => {
                option.classList.remove('active');
            });

            // Add active class to FacePay
            document.getElementById('facepayOption').classList.add('active');
            
            selectedPaymentMethod = 'facepay';
            
            // Show selected method
            const selectedMethodDiv = document.getElementById('selectedMethod');
            const selectedMethodText = document.getElementById('selectedMethodText');
            
            selectedMethodText.textContent = 'FacePay - Secure facial recognition payment via gateway';
            selectedMethodDiv.classList.add('show');
            
            // Enable proceed button
            document.getElementById('proceedBtn').disabled = false;
        }

        function showNotAvailable(methodName) {
            alert(`${methodName} is currently not available. Please use FacePay for payment.`);
        }

        function proceedWithPayment() {
            if (selectedPaymentMethod === 'facepay') {
                payWithGateway();
            } else {
                alert('Please select FacePay to proceed with payment.');
            }
        }

        function payWithGateway() {
            const subtotal = cart.reduce((sum, item) => sum + (item.price * item.quantity), 0);
            const tax = subtotal * 0.06;
            const total = subtotal + tax;
            const orderId = 'KIOSK-DEMO-' + Date.now();
            
            // Fill form data
            document.getElementById('gateway_amount').value = total.toFixed(2);
            document.getElementById('gateway_order_id').value = orderId;
            document.getElementById('gateway_description').value = `McDonald's Kiosk Order - ${cart.length} items`;
            
            // ✅ NEW: Submit to FacePay Gateway (which will handle success/failure and return here)
            console.log('Submitting to FacePay Gateway...', {
                amount: total.toFixed(2),
                orderId: orderId,
                merchant: '<?= MERCHANT_ID ?>'
            });
            
            document.getElementById('facepayGatewayForm').submit();
        }

        function goBack() {
            window.location.href = 'e-commerce.php';
        }
    </script>
</body>
</html>