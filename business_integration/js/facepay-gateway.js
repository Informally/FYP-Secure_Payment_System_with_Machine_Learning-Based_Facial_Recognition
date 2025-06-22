// business_integration/js/facepay-gateway.js
// Advanced API integration for kiosks (like Stripe.js)

class FacePayGateway {
    constructor(apiKey, options = {}) {
        this.apiKey = apiKey;
        this.baseUrl = options.baseUrl || 'http://localhost/payment_gateway/api';
        this.merchantId = options.merchantId || 'KIOSK001';
    }

    // Process payment using FacePay Gateway API
    async processPayment(paymentData) {
        try {
            const response = await fetch(`${this.baseUrl}/process_payment.php`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.apiKey}`
                },
                body: JSON.stringify({
                    amount: paymentData.amount,
                    order_id: paymentData.orderId,
                    customer_id: paymentData.customerId,
                    pin: paymentData.pin,
                    description: paymentData.description || '',
                    currency: paymentData.currency || 'MYR'
                })
            });

            const result = await response.json();
            
            if (response.ok && result.status === 'success') {
                return {
                    success: true,
                    data: result.data
                };
            } else {
                return {
                    success: false,
                    error: result.message,
                    code: response.status
                };
            }
        } catch (error) {
            return {
                success: false,
                error: 'Network error: ' + error.message
            };
        }
    }

    // Verify payment status
    async verifyPayment(transactionId) {
        try {
            const response = await fetch(
                `${this.baseUrl}/verify_payment.php?transaction_id=${transactionId}`,
                {
                    method: 'GET',
                    headers: {
                        'Authorization': `Bearer ${this.apiKey}`
                    }
                }
            );

            const result = await response.json();
            
            if (response.ok && result.status === 'success') {
                return {
                    success: true,
                    data: result.data
                };
            } else {
                return {
                    success: false,
                    error: result.message
                };
            }
        } catch (error) {
            return {
                success: false,
                error: 'Network error: ' + error.message
            };
        }
    }

    // Get customer balance (if customer provides their ID and PIN)
    async getCustomerBalance(customerId, pin) {
        try {
            const response = await fetch(`${this.baseUrl}/customer_balance.php`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.apiKey}`
                },
                body: JSON.stringify({
                    customer_id: customerId,
                    pin: pin
                })
            });

            const result = await response.json();
            
            if (response.ok && result.status === 'success') {
                return {
                    success: true,
                    balance: result.data.balance,
                    customerName: result.data.customer_name
                };
            } else {
                return {
                    success: false,
                    error: result.message
                };
            }
        } catch (error) {
            return {
                success: false,
                error: 'Network error: ' + error.message
            };
        }
    }
}

// Usage example in kiosk application
class KioskPaymentHandler {
    constructor() {
        // Initialize FacePay Gateway with merchant API key
        this.facePayGateway = new FacePayGateway('sk_kiosk_a1b2c3d4e5f6g7h8i9j0k1l2', {
            merchantId: 'KIOSK001'
        });
    }

    // Method 1: Direct API payment (customer enters their ID + PIN)
    async processDirectPayment(cart, customerId, pin) {
        const total = this.calculateTotal(cart);
        const orderId = 'KIOSK-' + Date.now();

        const paymentResult = await this.facePayGateway.processPayment({
            amount: total,
            orderId: orderId,
            customerId: customerId,
            pin: pin,
            description: `Kiosk Order - ${cart.length} items`
        });

        if (paymentResult.success) {
            // Payment successful
            this.showSuccessPage(paymentResult.data);
            this.clearCart();
        } else {
            // Payment failed
            this.showErrorPage(paymentResult.error);
        }

        return paymentResult;
    }

    // Method 2: Gateway redirect (recommended for kiosks)
    redirectToGateway(cart) {
        const total = this.calculateTotal(cart);
        const orderId = 'KIOSK-' + Date.now();

        // Create and submit gateway form (like PayPal checkout)
        const form = document.createElement('form');
        form.method = 'POST';
        form.action = 'http://localhost/payment_gateway/checkout.php';

        const fields = {
            merchant_id: 'KIOSK001',
            amount: total.toFixed(2),
            order_id: orderId,
            currency: 'MYR',
            return_url: 'http://localhost/business_integration/payment_success.php',
            cancel_url: 'http://localhost/business_integration/payment_failed.php',
            description: `Kiosk Order - ${cart.length} items`
        };

        for (const [key, value] of Object.entries(fields)) {
            const input = document.createElement('input');
            input.type = 'hidden';
            input.name = key;
            input.value = value;
            form.appendChild(input);
        }

        document.body.appendChild(form);
        form.submit();
    }

    // Check customer balance before payment
    async checkCustomerBalance(customerId, pin) {
        const balanceResult = await this.facePayGateway.getCustomerBalance(customerId, pin);
        
        if (balanceResult.success) {
            return {
                hasBalance: true,
                balance: balanceResult.balance,
                customerName: balanceResult.customerName
            };
        } else {
            return {
                hasBalance: false,
                error: balanceResult.error
            };
        }
    }

    // Verify payment after customer returns from gateway
    async verifyPaymentReturn(transactionId) {
        const verifyResult = await this.facePayGateway.verifyPayment(transactionId);
        
        if (verifyResult.success && verifyResult.data.status === 'success') {
            this.showSuccessPage(verifyResult.data);
            return true;
        } else {
            this.showErrorPage('Payment verification failed');
            return false;
        }
    }

    calculateTotal(cart) {
        const subtotal = cart.reduce((sum, item) => sum + (item.price * item.quantity), 0);
        const tax = subtotal * 0.06;
        return subtotal + tax;
    }

    showSuccessPage(transactionData) {
        window.location.href = `payment_success.php?transaction_id=${transactionData.transaction_id}&amount=${transactionData.amount}`;
    }

    showErrorPage(error) {
        window.location.href = `payment_failed.php?error=${encodeURIComponent(error)}`;
    }

    clearCart() {
        sessionStorage.removeItem('kioskCart');
        localStorage.clear();
    }
}

// Initialize payment handler when page loads
document.addEventListener('DOMContentLoaded', function() {
    window.kioskPayment = new KioskPaymentHandler();
});

// Example usage in your kiosk:
/*
// For gateway redirect (recommended):
document.getElementById('payWithFacePay').addEventListener('click', function() {
    window.kioskPayment.redirectToGateway(cart);
});

// For direct API payment (advanced):
document.getElementById('directPayment').addEventListener('click', async function() {
    const customerId = prompt('Enter your FacePay ID:');
    const pin = prompt('Enter your PIN:');
    
    if (customerId && pin) {
        await window.kioskPayment.processDirectPayment(cart, customerId, pin);
    }
});
*/