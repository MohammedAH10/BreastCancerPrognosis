import logging
from typing import Dict, Any
from .survival_predictor import SurvivalPredictor

logger = logging.getLogger(__name__)

class InsightEngine:
    def __init__(self):
        self.benign_patterns = self._load_benign_patterns()
        self.malignant_patterns = self._load_malignant_patterns()
        self.survival_predictor = SurvivalPredictor()
    
    def _load_benign_patterns(self):
        return {
            'risk_factors': {
                'family_history': {'weight': 2, 'message': 'Family history of breast cancer'},
                'age_50_plus': {'weight': 2, 'message': 'Age 50 or older'},
                'post_menopausal': {'weight': 1, 'message': 'Post-menopausal status'},
                'bmi_obese': {'weight': 1, 'message': 'Obesity (BMI ≥ 30)'},
            }
        }
    
    def _load_malignant_patterns(self):
        return {
            'stage_estimation': {
                'tumor_2cm_negative': 'Stage I',
                'tumor_2_5cm_negative': 'Stage II',
                'tumor_any_positive': 'Stage II-III',
            }
        }
    
    def generate_insights(self, prediction_type: str, user_responses: Dict[str, Any], image_confidence: float):
        """Generate insights based on prediction type"""
        if prediction_type == 'benign':
            return self._generate_benign_insights(user_responses, image_confidence)
        else:
            return self._generate_malignant_insights(user_responses, image_confidence)
    
    def _generate_benign_insights(self, user_responses: Dict[str, Any], image_confidence: float):
        """Generate insights for benign predictions"""
        try:
            risk_assessment = self._assess_benign_risk(user_responses, image_confidence)
            lifestyle_recommendations = self._get_benign_lifestyle_recommendations(user_responses)
            follow_up_plan = self._generate_benign_follow_up(risk_assessment['risk_level'])
            
            return {
                'type': 'benign',
                'risk_assessment': risk_assessment,
                'lifestyle_recommendations': lifestyle_recommendations,
                'follow_up_plan': follow_up_plan,
                'general_advice': [
                    "Continue monthly self-breast examinations",
                    "Maintain healthy body weight through balanced diet",
                    "Engage in regular physical activity (150 minutes/week)",
                    "Limit alcohol consumption to 1 drink per day or less",
                    "Avoid smoking and secondhand smoke exposure",
                    "Report any breast changes to your doctor immediately",
                    "Attend all scheduled screening appointments",
                    "Consider genetic counseling if strong family history"
                ]
            }
        except Exception as e:
            logger.error(f"Error generating benign insights: {e}")
            return self._get_fallback_insights('benign')
    
    def _generate_malignant_insights(self, user_responses: Dict[str, Any], image_confidence: float):
        """Generate insights for malignant predictions with survival prediction"""
        try:
            # Get survival prediction
            survival_features = self.survival_predictor.map_questionnaire_to_survival_features(user_responses)
            survival_prediction = self.survival_predictor.predict_survival(survival_features)
            
            clinical_insights = self._assess_malignant_clinical(user_responses, image_confidence)
            treatment_recommendations = self._get_malignant_treatments(clinical_insights['stage_estimate'])
            prognosis_indicators = self._assess_malignant_prognosis(user_responses, survival_prediction)
            
            return {
                'type': 'malignant',
                'clinical_insights': clinical_insights,
                'treatment_recommendations': treatment_recommendations,
                'prognosis_indicators': prognosis_indicators,
                'survival_prediction': survival_prediction,
                'next_steps': [
                    'Consult with breast surgeon and oncologist',
                    'Complete diagnostic imaging (MRI, ultrasound)',
                    'Schedule biopsy for confirmation',
                    'Multidisciplinary team evaluation',
                    f"Predicted 5-year survival: {survival_prediction['survival_percentage']}%"
                ]
            }
        except Exception as e:
            logger.error(f"Error generating malignant insights: {e}")
            return self._get_fallback_insights('malignant')
    
    def _assess_benign_risk(self, responses, image_confidence):
        risk_score = 0
        risk_factors = []
        
        if responses.get('family_history') == 'Yes':
            risk_score += 2
            risk_factors.append("Family history of breast cancer")
        
        age_group = responses.get('age_group', '')
        if any(age in age_group for age in ['50-59', '60-69', '70 and above']):
            risk_score += 2
            risk_factors.append(f"Age group: {age_group}")
        
        if responses.get('menopausal_status') == 'Post-menopausal':
            risk_score += 1
            risk_factors.append("Post-menopausal status")
        
        bmi_category = responses.get('bmi_category', '')
        if '30 and above' in bmi_category:
            risk_score += 1
            risk_factors.append("Obesity (BMI ≥ 30)")
        elif '25-29.9' in bmi_category:
            risk_score += 0.5
            risk_factors.append("Overweight (BMI 25-29.9)")
        
        # Determine risk level
        if risk_score >= 3:
            risk_level = "High"
            risk_color = "red"
        elif risk_score >= 2:
            risk_level = "Moderate"
            risk_color = "orange"
        else:
            risk_level = "Low"
            risk_color = "green"
        
        return {
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'image_confidence': f"{image_confidence:.1%}"
        }
    
    def _get_benign_lifestyle_recommendations(self, responses):
        recommendations = []
        
        if responses.get('smoking') == 'Yes':
            recommendations.append("Smoking cessation program recommended")
        
        if responses.get('alcohol') == 'Yes':
            recommendations.append("Limit alcohol to 1 drink per day or less")
        
        bmi = responses.get('bmi_category', '')
        if '30 and above' in bmi:
            recommendations.extend([
                "Weight management through balanced diet",
                "Regular exercise program (150+ minutes/week)",
                "Consult nutritionist for dietary planning"
            ])
        elif '25-29.9' in bmi:
            recommendations.extend([
                "Moderate weight loss recommended",
                "Increase physical activity level",
                "Focus on whole foods and portion control"
            ])
        
        if responses.get('physical_activity') == 'Sedentary (little or no exercise)':
            recommendations.extend([
                "Gradually increase physical activity to 150 minutes/week",
                "Consider walking, swimming, or cycling",
                "Incorporate strength training 2 times/week"
            ])
        
        # Add general healthy lifestyle recommendations
        general_recommendations = [
            "Maintain balanced diet rich in fruits and vegetables",
            "Include lean proteins and whole grains in diet",
            "Practice stress management techniques",
            "Get adequate sleep (7-9 hours per night)",
            "Stay hydrated with water throughout the day"
        ]
        
        recommendations.extend(general_recommendations)
        return recommendations if recommendations else ["Maintain current healthy lifestyle habits"]
    
    def _generate_benign_follow_up(self, risk_level):
        if risk_level == "High":
            return {
                'timeline': '6-month follow-up recommended',
                'recommendations': [
                    'Clinical breast exam in 6 months',
                    'Diagnostic mammogram',
                    'Consider breast MRI if dense tissue',
                    'Regular self-breast exams monthly'
                ]
            }
        elif risk_level == "Moderate":
            return {
                'timeline': 'Annual screening recommended',
                'recommendations': [
                    'Clinical breast exam annually',
                    'Screening mammogram yearly',
                    'Monthly self-breast exams',
                    'Maintain healthy lifestyle'
                ]
            }
        else:
            return {
                'timeline': 'Routine screening schedule',
                'recommendations': [
                    'Annual screening mammogram',
                    'Clinical breast exam every 1-2 years',
                    'Monthly self-breast exams',
                    'Continue healthy habits'
                ]
            }
    
    def _assess_malignant_clinical(self, responses, image_confidence):
        tumor_size = responses.get('tumor_size', 'Not sure')
        lymph_nodes = responses.get('lymph_node_status', 'Not sure')
        
        if tumor_size == 'Less than 2 cm' and lymph_nodes == 'Negative':
            stage = 'Likely Stage I'
            stage_color = 'green'
        elif tumor_size == '2-5 cm' and lymph_nodes == 'Negative':
            stage = 'Likely Stage II'
            stage_color = 'yellow'
        elif lymph_nodes == 'Positive':
            stage = 'Likely Stage II-III'
            stage_color = 'orange'
        else:
            stage = 'Clinical assessment needed'
            stage_color = 'gray'
        
        return {
            'stage_estimate': stage,
            'stage_color': stage_color,
            'image_confidence': f"{image_confidence:.1%}",
            'next_diagnostics': ['Confirmatory biopsy', 'Breast MRI', 'Complete staging workup']
        }
    
    def _get_malignant_treatments(self, stage_estimate):
        if 'Stage I' in stage_estimate:
            return [
                'Surgery (Lumpectomy or Mastectomy)',
                'Radiation therapy typically after lumpectomy',
                'Possible chemotherapy based on tumor characteristics',
                'Hormone therapy if hormone receptor positive'
            ]
        elif 'Stage II' in stage_estimate:
            return [
                'Surgery (Lumpectomy with radiation or Mastectomy)',
                'Chemotherapy typically recommended',
                'Radiation therapy if lumpectomy performed',
                'Targeted therapy if HER2 positive',
                'Hormone therapy if hormone receptor positive'
            ]
        else:
            return [
                'Neoadjuvant chemotherapy to shrink tumor before surgery',
                'Surgery (Mastectomy typically recommended)',
                'Radiation therapy after surgery',
                'Adjuvant systemic therapy',
                'Targeted therapies based on biomarker testing'
            ]
    
    def _assess_malignant_prognosis(self, responses, survival_prediction):
        positive_factors = []
        
        if responses.get('family_history') == 'No':
            positive_factors.append('No family history of breast cancer')
        
        age_group = responses.get('age_group', '')
        if any(age in age_group for age in ['30-39', '40-49']):
            positive_factors.append('Younger age group (better treatment tolerance)')
        
        if responses.get('symptoms_duration') == 'Less than 1 month':
            positive_factors.append('Early symptom recognition')
        
        tumor_size = responses.get('tumor_size', '')
        if 'Less than 2 cm' in tumor_size:
            positive_factors.append('Small tumor size')
        
        if responses.get('lymph_node_status') == 'Negative':
            positive_factors.append('No lymph node involvement indicated')
        
        return {
            'positive_factors': positive_factors,
            'survival_percentage': survival_prediction['survival_percentage'],
            'survival_confidence': survival_prediction['confidence'],
            'survival_message': survival_prediction['message'],
            'recommendation': 'Early detection and comprehensive treatment significantly improve outcomes'
        }
    
    def _get_fallback_insights(self, prediction_type):
        """Provide fallback insights in case of errors"""
        if prediction_type == 'benign':
            return {
                'type': 'benign',
                'risk_assessment': {
                    'risk_level': 'Moderate',
                    'risk_color': 'orange',
                    'risk_score': 2,
                    'risk_factors': ['Standard risk assessment'],
                    'image_confidence': '85.0%'
                },
                'lifestyle_recommendations': [
                    'Maintain healthy lifestyle with balanced diet',
                    'Regular exercise (150 minutes/week)',
                    'Limit alcohol consumption',
                    'Avoid tobacco products',
                    'Monthly self-breast exams'
                ],
                'follow_up_plan': {
                    'timeline': 'Annual screening recommended',
                    'recommendations': ['Clinical breast exam', 'Screening mammogram']
                },
                'general_advice': [
                    'Consult with healthcare provider for personalized medical advice',
                    'Maintain regular screening schedule',
                    'Report any changes in breast tissue promptly'
                ]
            }
        else:
            return {
                'type': 'malignant',
                'clinical_insights': {
                    'stage_estimate': 'Clinical assessment needed',
                    'stage_color': 'gray',
                    'image_confidence': '90.0%',
                    'next_diagnostics': ['Consult with specialist for complete evaluation']
                },
                'treatment_recommendations': ['Specialist consultation required for treatment planning'],
                'prognosis_indicators': {
                    'positive_factors': ['Early detection improves outcomes'],
                    'survival_percentage': 75.0,
                    'survival_confidence': 'medium',
                    'survival_message': 'Comprehensive evaluation needed for accurate prognosis',
                    'recommendation': 'Comprehensive medical evaluation needed'
                },
                'next_steps': ['Consult with oncologist', 'Complete diagnostic workup', 'Multidisciplinary team evaluation']
            }