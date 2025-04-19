document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const form = document.getElementById('prediction-form');
    const resultsContainer = document.getElementById('results');
    const deficitValue = document.getElementById('deficit-value');
    const deficitPercentage = document.getElementById('deficit-percentage');
    
    // Slider value display
    const tariffsSlider = document.getElementById('tariffs');
    const tariffsValue = document.getElementById('tariffs-value');
    const responseSlider = document.getElementById('response');
    const responseValue = document.getElementById('response-value');
    
    // Update slider value displays
    tariffsSlider.addEventListener('input', function() {
        tariffsValue.textContent = this.value + '%';
    });
    
    responseSlider.addEventListener('input', function() {
        responseValue.textContent = this.value + '%';
    });
    
    // Chart initialization
    let predictionChart = null;
    
    // Form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Get form values
        const country = document.getElementById('country').value;
        const exports = parseFloat(document.getElementById('exports').value);
        const imports = parseFloat(document.getElementById('imports').value);
        const tariffs = parseFloat(document.getElementById('tariffs').value) / 100;
        const response = parseFloat(document.getElementById('response').value) / 100;
        const population = parseInt(document.getElementById('population').value);
        
        try {
            // Call the API
            const result = await fetchPrediction(exports, imports, tariffs, response, population);
            
            // Display results
            deficitValue.textContent = formatCurrency(result.prediction);
            deficitPercentage.textContent = `${result.percentage.toFixed(2)}% of imports`;
            
            // Show results container
            resultsContainer.classList.remove('hidden');
            
            // Update chart
            updateChart(country, exports, imports, result.prediction);
            
            // Scroll to results
            resultsContainer.scrollIntoView({ behavior: 'smooth' });
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while making the prediction. Please try again.');
        }
    });
    
    // Fetch prediction from API
    async function fetchPrediction(exports, imports, tariffs, response, population) {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                exports,
                imports,
                tariffs,
                response,
                population
            }),
        });
        
        if (!response.ok) {
            throw new Error('API request failed');
        }
        
        return await response.json();
    }
    
    // Format currency
    function formatCurrency(value) {
        return new Intl.NumberFormat('en-US', { 
            style: 'currency', 
            currency: 'USD',
            maximumFractionDigits: 2
        }).format(value);
    }
    
    // Update chart
    function updateChart(country, exports, imports, deficit) {
        const ctx = document.getElementById('prediction-chart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (predictionChart) {
            predictionChart.destroy();
        }
        
        // Create new chart
        predictionChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Exports', 'Imports', 'Trade Deficit'],
                datasets: [{
                    label: `${country} Trade Data`,
                    data: [exports, imports, Math.abs(deficit)],
                    backgroundColor: [
                        'rgba(46, 204, 113, 0.7)',
                        'rgba(231, 76, 60, 0.7)',
                        deficit < 0 ? 'rgba(52, 152, 219, 0.7)' : 'rgba(155, 89, 182, 0.7)'
                    ],
                    borderColor: [
                        'rgba(46, 204, 113, 1)',
                        'rgba(231, 76, 60, 1)',
                        deficit < 0 ? 'rgba(52, 152, 219, 1)' : 'rgba(155, 89, 182, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'USD'
                        }
                    }
                }
            }
        });
    }
});