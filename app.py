from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import uuid
from datetime import datetime
from services.ai_service import AIService
from services.insight_engine import InsightEngine
from utils.image_processing import save_uploaded_image, allowed_file

# Initialize Flask app
app = Flask(__name__)
app.config.from_object('config.Config')

# Initialize services
ai_service = AIService(app.config['MODEL_PATH'])
insight_engine = InsightEngine()

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Questionnaire templates
QUESTIONNAIRES = {
    'benign': {
        'title': 'Benign Case Assessment',
        'sections': {
            'demographics': {
                'title': 'Demographic Information',
                'questions': {
                    'age_group': {
                        'question': 'What is your age group?',
                        'type': 'select',
                        'options': ['Under 30', '30-39', '40-49', '50-59', '60-69', '70 and above'],
                        'required': True
                    },
                    'family_history': {
                        'question': 'Do you have a family history of breast cancer?',
                        'type': 'radio',
                        'options': ['Yes', 'No'],
                        'required': True
                    }
                }
            },
            'lifestyle': {
                'title': 'Lifestyle Factors',
                'questions': {
                    'bmi_category': {
                        'question': 'What is your BMI category?',
                        'type': 'select',
                        'options': [
                            'Below 18.5 (Underweight)',
                            '18.5-24.9 (Normal weight)',
                            '25-29.9 (Overweight)',
                            '30 and above (Obese)'
                        ],
                        'required': True
                    },
                    'smoking': {
                        'question': 'Do you smoke?',
                        'type': 'radio',
                        'options': ['Yes', 'No'],
                        'required': True
                    },
                    'alcohol': {
                        'question': 'Do you consume alcohol?',
                        'type': 'radio',
                        'options': ['Yes', 'No'],
                        'required': True
                    },
                    'physical_activity': {
                        'question': 'What is your physical activity level?',
                        'type': 'select',
                        'options': [
                            'Sedentary (little or no exercise)',
                            'Lightly active (light exercise/sports 1-3 days/week)',
                            'Moderately active (moderate exercise/sports 3-5 days/week)',
                            'Very active (hard exercise/sports 6-7 days a week)'
                        ],
                        'required': True
                    }
                }
            },
            'health_history': {
                'title': 'Health History',
                'questions': {
                    'menopausal_status': {
                        'question': 'What is your menopausal status?',
                        'type': 'select',
                        'options': ['Pre-menopausal', 'Peri-menopausal', 'Post-menopausal'],
                        'required': True
                    },
                    'previous_biopsy': {
                        'question': 'Have you had a previous breast biopsy?',
                        'type': 'radio',
                        'options': ['Yes', 'No'],
                        'required': True
                    }
                }
            }
        }
    },
    'malignant': {
        'title': 'Comprehensive Clinical Assessment',
        'sections': {
            'demographics': {
                'title': 'Demographic Information',
                'questions': {
                    'age_group': {
                        'question': 'What is your age group?',
                        'type': 'select',
                        'options': ['Under 30', '30-39', '40-49', '50-59', '60-69', '70 and above'],
                        'required': True
                    },
                    'ethnicity': {
                        'question': 'What is your ethnicity?',
                        'type': 'select',
                        'options': ['Hausa', 'Yoruba', 'Igbo', 'Other Nigerian ethnicity', 'Other African', 'Other'],
                        'required': True
                    },
                    'marital_status': {
                        'question': 'What is your marital status?',
                        'type': 'select',
                        'options': ['Single', 'Married', 'Divorced', 'Widowed'],
                        'required': True
                    },
                    'family_history': {
                        'question': 'Do you have a family history of breast cancer?',
                        'type': 'radio',
                        'options': ['Yes', 'No'],
                        'required': True
                    }
                }
            },
            'clinical_details': {
                'title': 'Clinical Details',
                'questions': {
                    'menopausal_status': {
                        'question': 'What is your menopausal status?',
                        'type': 'select',
                        'options': ['Pre-menopausal', 'Peri-menopausal', 'Post-menopausal'],
                        'required': True
                    },
                    'cancer_stage': {
                        'question': 'What is the stage of breast cancer at diagnosis?',
                        'type': 'select',
                        'options': ['Stage I', 'Stage II', 'Stage III', 'Stage IV', 'Not sure'],
                        'required': True
                    },
                    'tumor_size': {
                        'question': 'What is the tumor size?',
                        'type': 'select',
                        'options': ['Less than 2 cm', '2-5 cm', 'Greater than 5 cm', 'Not sure'],
                        'required': True
                    },
                    'tumor_type': {
                        'question': 'What is the tumor type?',
                        'type': 'select',
                        'options': [
                            'Invasive Ductal Carcinoma (IDC)',
                            'Invasive Lobular Carcinoma (ILC)',
                            'Ductal Carcinoma In Situ (DCIS)',
                            'Other',
                            'Not sure'
                        ],
                        'required': True
                    },
                    'tumor_grade': {
                        'question': 'What is the tumor grade?',
                        'type': 'select',
                        'options': [
                            'Grade 1 (Well-differentiated)',
                            'Grade 2 (Moderately differentiated)',
                            'Grade 3 (Poorly differentiated)',
                            'Not sure'
                        ],
                        'required': True
                    },
                    'lymph_node_status': {
                        'question': 'What is the lymph node status?',
                        'type': 'select',
                        'options': ['Positive', 'Negative', 'Not sure'],
                        'required': True
                    }
                }
            },
            'biomarkers': {
                'title': 'Biomarker Status',
                'questions': {
                    'er_status': {
                        'question': 'Estrogen Receptor (ER) Status',
                        'type': 'select',
                        'options': ['Positive', 'Negative', 'Not sure'],
                        'required': True
                    },
                    'pr_status': {
                        'question': 'Progesterone Receptor (PR) Status',
                        'type': 'select',
                        'options': ['Positive', 'Negative', 'Not sure'],
                        'required': True
                    },
                    'her2_status': {
                        'question': 'HER2 Status',
                        'type': 'select',
                        'options': ['Positive', 'Negative', 'Not sure'],
                        'required': True
                    }
                }
            },
            'treatment_history': {
                'title': 'Treatment History',
                'questions': {
                    'treatment_types': {
                        'question': 'What types of treatment have you received?',
                        'type': 'select',
                        'options': [
                            'Surgery',
                            'Chemotherapy', 
                            'Radiation therapy',
                            'Hormone therapy',
                            'Targeted therapy',
                            'Not started treatment yet'
                        ],
                        'required': True
                    },
                    'treatment_duration': {
                        'question': 'Duration of treatment',
                        'type': 'select',
                        'options': [
                            'Less than 3 months',
                            '3-6 months',
                            '6-12 months',
                            'Over 12 months',
                            'Not applicable'
                        ],
                        'required': True
                    },
                    'treatment_status': {
                        'question': 'Current treatment status',
                        'type': 'select',
                        'options': ['Ongoing', 'Completed', 'Not started'],
                        'required': True
                    },
                    'diagnosis_to_treatment': {
                        'question': 'Time from diagnosis to treatment start',
                        'type': 'select',
                        'options': ['Less than 1 month', '1-3 months', '3-6 months', 'Over 6 months'],
                        'required': True
                    }
                }
            },
            'health_status': {
                'title': 'Current Health Status',
                'questions': {
                    'recurrence': {
                        'question': 'Have you experienced a recurrence of breast cancer?',
                        'type': 'radio',
                        'options': ['Yes', 'No'],
                        'required': True
                    },
                    'follow_up_frequency': {
                        'question': 'How often do you attend follow-up appointments?',
                        'type': 'select',
                        'options': [
                            'Every month',
                            'Every 3 months', 
                            'Every 6 months',
                            'Yearly',
                            'Not regularly'
                        ],
                        'required': True
                    },
                    'bmi_category': {
                        'question': 'What is your BMI category?',
                        'type': 'select',
                        'options': [
                            'Below 18.5 (Underweight)',
                            '18.5-24.9 (Normal weight)',
                            '25-29.9 (Overweight)',
                            '30 and above (Obese)',
                            'Not sure'
                        ],
                        'required': True
                    },
                    'current_health': {
                        'question': 'Current health status (self-assessed)',
                        'type': 'select',
                        'options': ['Excellent', 'Good', 'Fair', 'Poor'],
                        'required': True
                    },
                    'other_conditions': {
                        'question': 'Other chronic conditions',
                        'type': 'select',
                        'options': [
                            'None',
                            'Hypertension',
                            'Diabetes',
                            'Heart disease',
                            'Other'
                        ],
                        'required': True
                    }
                }
            }
        }
    }
}

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Image upload and prediction page"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'image' not in request.files:
            return render_template('prediction.html', error='No file selected')
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return render_template('prediction.html', error='No file selected')
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return render_template('prediction.html', error='Invalid file type. Please upload PNG, JPG, or JPEG.')
        
        try:
            # Save uploaded image
            filename = save_uploaded_image(file, app.config['UPLOAD_FOLDER'])
            file_path = f"uploads/{filename}"
            
            # Make prediction
            prediction_result = ai_service.predict(file)
            
            # Store in session
            session['prediction'] = prediction_result
            session['image_path'] = file_path
            session['prediction_id'] = str(uuid.uuid4())
            
            # Redirect to questionnaire
            return redirect(url_for('questionnaire'))
            
        except Exception as e:
            return render_template('prediction.html', error=f'Error processing image: {str(e)}')
    
    return render_template('prediction.html')

@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    """Dynamic questionnaire based on prediction"""
    # Check if prediction exists in session
    if 'prediction' not in session:
        return redirect(url_for('predict'))
    
    prediction_type = session['prediction']['result']
    questionnaire_data = QUESTIONNAIRES.get(prediction_type)
    
    if not questionnaire_data:
        return redirect(url_for('predict'))
    
    if request.method == 'POST':
        # Collect form responses
        responses = {}
        for section_name, section in questionnaire_data['sections'].items():
            for question_name in section['questions']:
                responses[question_name] = request.form.get(question_name, '')
        
        # Store responses in session
        session['responses'] = responses
        
        # Generate insights using the updated insight engine
        insights = insight_engine.generate_insights(
            prediction_type, 
            responses, 
            session['prediction']['confidence']
        )
        session['insights'] = insights
        
        return redirect(url_for('insights'))
    
    return render_template('questionnaire.html', 
                            questionnaire=questionnaire_data,
                            prediction_type=prediction_type,
                            prediction=session['prediction'])

@app.route('/insights')
def insights():
    """Display insights and recommendations"""
    if 'insights' not in session:
        return redirect(url_for('questionnaire'))
    
    return render_template('insights.html',
                            prediction=session['prediction'],
                            insights=session['insights'],
                            image_path=session.get('image_path'),
                            responses=session.get('responses', {}))

@app.route('/new-analysis')
def new_analysis():
    """Reset session and start new analysis"""
    session.clear()
    return redirect(url_for('predict'))

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    })

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error='Internal server error'), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)