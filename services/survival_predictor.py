import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)

class SurvivalPredictor:
    def __init__(self, model_path='breast_cancer_survival_predictor.pkl'):
        self.model_path = model_path
        self.model_data = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained survival prediction model"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Survival model file not found: {self.model_path}. Using mock predictions.")
                return
            
            self.model_data = joblib.load(self.model_path)
            logger.info("Survival prediction model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading survival model: {e}")
            self.model_data = None
    
    def predict_survival(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict survival percentage for a patient"""
        if self.model_data is None:
            return self._mock_survival_prediction(patient_data)
        
        try:
            model = self.model_data['model']
            label_encoders = self.model_data['label_encoders']
            feature_columns = self.model_data['feature_columns']
            
            # Convert patient data to dataframe
            patient_df = pd.DataFrame([patient_data])
            
            # Preprocess using the same encoders
            for col in patient_df.columns:
                if col in label_encoders:
                    try:
                        patient_df[col] = label_encoders[col].transform(patient_df[col])
                    except ValueError:
                        # If label not seen during training, use 'Unknown' encoding
                        patient_df[col] = label_encoders[col].transform(['Unknown'])[0]
            
            # Ensure all feature columns are present
            for col in feature_columns:
                if col not in patient_df.columns:
                    patient_df[col] = 0  # Default value for missing columns
            
            # Reorder columns to match training
            patient_df = patient_df[feature_columns]
            
            # Predict
            survival_percentage = model.predict(patient_df)[0]
            survival_percentage = max(0, min(100, survival_percentage))
            
            return {
                'survival_percentage': round(survival_percentage, 1),
                'confidence': 'high' if survival_percentage > 70 else 'medium' if survival_percentage > 50 else 'low',
                'message': self._get_survival_message(survival_percentage),
                'is_mock': False
            }
            
        except Exception as e:
            logger.error(f"Error in survival prediction: {e}")
            return self._mock_survival_prediction(patient_data)
    
    def _mock_survival_prediction(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide mock survival prediction when model is not available"""
        import random
        
        # Simple heuristic based on available data
        base_survival = 75
        
        # Adjust based on stage if available
        stage = patient_data.get('Stage of Breast Cancer at Diagnosis (if applicable): ', 'Unknown')
        stage_adjustments = {
            'Stage I': 15,
            'Stage II': 0,
            'Stage III': -20,
            'Stage IV': -40
        }
        
        adjustment = stage_adjustments.get(stage, 0)
        survival_percentage = base_survival + adjustment + random.uniform(-5, 5)
        survival_percentage = max(10, min(95, survival_percentage))
        
        return {
            'survival_percentage': round(survival_percentage, 1),
            'confidence': 'medium',
            'message': self._get_survival_message(survival_percentage),
            'is_mock': True
        }
    
    def _get_survival_message(self, survival_percentage: float) -> str:
        """Get appropriate message based on survival percentage"""
        if survival_percentage >= 80:
            return "Excellent prognosis with current treatment approaches"
        elif survival_percentage >= 60:
            return "Good prognosis with comprehensive treatment plan"
        elif survival_percentage >= 40:
            return "Moderate prognosis - aggressive treatment recommended"
        else:
            return "Requires intensive treatment and close monitoring"
    
    def map_questionnaire_to_survival_features(self, questionnaire_responses: Dict[str, Any]) -> Dict[str, Any]:
        """Map questionnaire responses to survival model features with exact field names"""
        mapped_data = {}
        
        # Age mapping - exact field name match
        age_mapping = {
            'Under 30': 'Under 30',
            '30-39': '30-39', 
            '40-49': '40-49',
            '50-59': '50-59',
            '60-69': '60-69',
            '70 and above': '70 and above'
        }
        mapped_data['Age (in years): '] = age_mapping.get(
            questionnaire_responses.get('age_group', '40-49'), '40-49'
        )
        
        # Ethnicity
        mapped_data['Ethnicity'] = questionnaire_responses.get('ethnicity', 'Unknown')
        
        # Marital Status
        mapped_data['Marital Status'] = questionnaire_responses.get('marital_status', 'Unknown')
        
        # Family history - exact field name match
        mapped_data['Family History of Breast Cancer: '] = questionnaire_responses.get('family_history', 'No')
        
        # Menopausal status - exact field name match
        menopausal_mapping = {
            'Pre-menopausal': 'Pre-menopausal',
            'Peri-menopausal': 'Peri-menopausal', 
            'Post-menopausal': 'Post-menopausal'
        }
        mapped_data['Menopausal Status: '] = menopausal_mapping.get(
            questionnaire_responses.get('menopausal_status', 'Pre-menopausal'), 'Pre-menopausal'
        )
        
        # Other chronic conditions - exact field name match
        other_conditions = questionnaire_responses.get('other_conditions', 'None')
        if other_conditions == 'None':
            mapped_data['Other Chronic Conditions (check all that apply): '] = 'None'
        else:
            mapped_data['Other Chronic Conditions (check all that apply): '] = other_conditions
        
        # Cancer stage - exact field name match
        stage_mapping = {
            'Stage I': 'Stage I',
            'Stage II': 'Stage II', 
            'Stage III': 'Stage III',
            'Stage IV': 'Stage IV',
            'Not sure': 'Stage II'  # Default to Stage II if unsure
        }
        mapped_data['Stage of Breast Cancer at Diagnosis (if applicable): '] = stage_mapping.get(
            questionnaire_responses.get('cancer_stage', 'Not sure'), 'Stage II'
        )
        
        # Tumor size - exact field name match
        tumor_size_mapping = {
            'Not sure': '2-5 cm',
            'Less than 2 cm': 'Less than 2 cm',
            '2-5 cm': '2-5 cm',
            'Greater than 5 cm': 'Greater than 5 cm'
        }
        mapped_data['Tumor Size: '] = tumor_size_mapping.get(
            questionnaire_responses.get('tumor_size', 'Not sure'), '2-5 cm'
        )
        
        # Tumor type - exact field name match
        tumor_type_mapping = {
            'Invasive Ductal Carcinoma (IDC)': 'Invasive Ductal Carcinoma (IDC)',
            'Invasive Lobular Carcinoma (ILC)': 'Invasive Lobular Carcinoma (ILC)',
            'Ductal Carcinoma In Situ (DCIS)': 'Ductal Carcinoma In Situ (DCIS)',
            'Other': 'Other',
            'Not sure': 'Invasive Ductal Carcinoma (IDC)'  # Most common type
        }
        mapped_data['Tumor Type: '] = tumor_type_mapping.get(
            questionnaire_responses.get('tumor_type', 'Not sure'), 'Invasive Ductal Carcinoma (IDC)'
        )
        
        # Tumor grade - exact field name match
        tumor_grade_mapping = {
            'Grade 1 (Well-differentiated)': 'Grade 1 (Well-differentiated)',
            'Grade 2 (Moderately differentiated)': 'Grade 2 (Moderately differentiated)',
            'Grade 3 (Poorly differentiated)': 'Grade 3 (Poorly differentiated)',
            'Not sure': 'Grade 2 (Moderately differentiated)'
        }
        mapped_data['Tumor Grade: '] = tumor_grade_mapping.get(
            questionnaire_responses.get('tumor_grade', 'Not sure'), 'Grade 2 (Moderately differentiated)'
        )
        
        # Tumor stage (same as cancer stage)
        mapped_data['Tumor Stage: '] = stage_mapping.get(
            questionnaire_responses.get('cancer_stage', 'Not sure'), 'Stage II'
        )
        
        # Lymph node status - exact field name match
        lymph_mapping = {
            'Not sure': 'Unknown',
            'Positive': 'Positive',
            'Negative': 'Negative'
        }
        mapped_data['Lymph Node Status: '] = lymph_mapping.get(
            questionnaire_responses.get('lymph_node_status', 'Not sure'), 'Unknown'
        )
        
        # Receptor statuses - exact field name matches
        receptor_mapping = {
            'Positive': 'Positive',
            'Negative': 'Negative', 
            'Not sure': 'Unknown'
        }
        mapped_data['Estrogen Receptor (ER) Status: '] = receptor_mapping.get(
            questionnaire_responses.get('er_status', 'Not sure'), 'Unknown'
        )
        mapped_data['Progesterone Receptor (PR) Status: '] = receptor_mapping.get(
            questionnaire_responses.get('pr_status', 'Not sure'), 'Unknown'
        )
        mapped_data['Human Epidermal Growth Factor Receptor 2 (HER2) Status: '] = receptor_mapping.get(
            questionnaire_responses.get('her2_status', 'Not sure'), 'Unknown'
        )
        
        # Diagnosis to treatment - exact field name match
        diagnosis_mapping = {
            'Less than 1 month': 'Less than 1 month',
            '1-3 months': '1-3 months',
            '3-6 months': '3-6 months', 
            'Over 6 months': 'Over 6 months'
        }
        mapped_data['Duration from Diagnosis to Commencement of Treatment: '] = diagnosis_mapping.get(
            questionnaire_responses.get('diagnosis_to_treatment', '1-3 months'), '1-3 months'
        )
        
        # Recurrence - exact field name match
        mapped_data['Have you experienced a recurrence of breast cancer? '] = questionnaire_responses.get('recurrence', 'No')
        
        # Treatment types - exact field name match
        treatment_mapping = {
            'Surgery': 'Surgery',
            'Chemotherapy': 'Chemotherapy',
            'Radiation therapy': 'Radiation therapy', 
            'Hormone therapy': 'Hormone therapy',
            'Targeted therapy': 'Targeted therapy',
            'Not started treatment yet': 'None'
        }
        treatment_type = questionnaire_responses.get('treatment_types', 'Not started treatment yet')
        mapped_data['Type of Treatment Received (check all that apply): '] = treatment_mapping.get(treatment_type, 'None')
        
        # Treatment duration - exact field name match
        duration_mapping = {
            'Less than 3 months': 'Less than 3 months',
            '3-6 months': '3-6 months',
            '6-12 months': '6-12 months',
            'Over 12 months': 'Over 12 months',
            'Not applicable': 'Not applicable'
        }
        mapped_data['Duration of Treatment: '] = duration_mapping.get(
            questionnaire_responses.get('treatment_duration', 'Not applicable'), 'Not applicable'
        )
        
        # Treatment status - exact field name match
        status_mapping = {
            'Ongoing': 'Ongoing',
            'Completed': 'Completed',
            'Not started': 'Not started'
        }
        mapped_data['Current Treatment Status: '] = status_mapping.get(
            questionnaire_responses.get('treatment_status', 'Not started'), 'Not started'
        )
        
        # Follow-up frequency - exact field name match
        followup_mapping = {
            'Every month': 'Every month',
            'Every 3 months': 'Every 3 months',
            'Every 6 months': 'Every 6 months',
            'Yearly': 'Yearly',
            'Not regularly': 'Not regularly'
        }
        mapped_data['How often do you attend follow-up appointments? '] = followup_mapping.get(
            questionnaire_responses.get('follow_up_frequency', 'Every 3 months'), 'Every 3 months'
        )
        
        # Lifestyle factors
        mapped_data['Do you smoke? '] = questionnaire_responses.get('smoking', 'No')
        mapped_data['Do you consume alcohol? '] = questionnaire_responses.get('alcohol', 'No')
        
        # Physical activity - exact field name match
        activity_mapping = {
            'Sedentary (little or no exercise)': 'Sedentary (little or no exercise)',
            'Lightly active (light exercise/sports 1-3 days/week)': 'Lightly active (light exercise/sports 1-3 days/week)',
            'Moderately active (moderate exercise/sports 3-5 days/week)': 'Moderately active (moderate exercise/sports 3-5 days/week)',
            'Very active (hard exercise/sports 6-7 days a week)': 'Very active (hard exercise/sports 6-7 days a week)'
        }
        mapped_data['Physical Activity Level: '] = activity_mapping.get(
            questionnaire_responses.get('physical_activity', 'Moderately active (moderate exercise/sports 3-5 days/week)'),
            'Moderately active (moderate exercise/sports 3-5 days/week)'
        )
        
        # BMI - exact field name match
        bmi_mapping = {
            'Below 18.5 (Underweight)': 'Below 18.5 (Underweight)',
            '18.5-24.9 (Normal weight)': '18.5-24.9 (Normal weight)',
            '25-29.9 (Overweight)': '25-29.9 (Overweight)',
            '30 and above (Obese)': '30 and above (Obese)',
            'Not sure': '18.5-24.9 (Normal weight)'
        }
        mapped_data['Body Mass Index (BMI) (if known):'] = bmi_mapping.get(
            questionnaire_responses.get('bmi_category', 'Not sure'), '18.5-24.9 (Normal weight)'
        )
        
        # Current health - exact field name match
        health_mapping = {
            'Excellent': 'Excellent',
            'Good': 'Good',
            'Fair': 'Fair',
            'Poor': 'Poor'
        }
        mapped_data['Current Health Status (self-assessed): '] = health_mapping.get(
            questionnaire_responses.get('current_health', 'Good'), 'Good'
        )
        
        return mapped_data