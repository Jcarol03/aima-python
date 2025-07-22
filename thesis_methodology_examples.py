"""
Practical Examples for Financial Time Series Thesis Methodology
================================================================

This module provides concrete examples and code templates for implementing
the methodology outlined in the thesis framework.

Focus Areas:
1. Data preprocessing and feature engineering
2. Model implementation examples
3. Evaluation frameworks
4. Visualization tools for thesis presentation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom framework
from financial_thesis_framework import (
    ThesisFramework, ExtremeVolatilityDetector, 
    ModelComparison, SentimentAnalyzer, generate_sample_data
)


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for financial time series thesis.
    """
    
    def __init__(self, lookback_window: int = 252):
        """
        Initialize data preprocessor.
        
        Parameters:
        -----------
        lookback_window : int
            Window size for rolling calculations (default: 252 trading days)
        """
        self.lookback_window = lookback_window
        self.scalers = {}
        
    def create_financial_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive financial features for modeling.
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            DataFrame with OHLCV data
            
        Returns:
        --------
        DataFrame with engineered features
        """
        df = price_data.copy()
        
        # Basic returns
        df['returns'] = df['price'].pct_change()
        df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
        
        # Volatility measures
        df['realized_vol'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        df['parkinson_vol'] = self._parkinson_volatility(df)
        df['garman_klass_vol'] = self._garman_klass_volatility(df)
        
        # Price-based features
        df['rsi'] = self._calculate_rsi(df['price'])
        df['bollinger_upper'], df['bollinger_lower'] = self._bollinger_bands(df['price'])
        df['price_sma_20'] = df['price'].rolling(window=20).mean()
        df['price_sma_50'] = df['price'].rolling(window=50).mean()
        
        # Volume features (if available)
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['price_volume'] = df['returns'] * df['volume']
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'vol_lag_{lag}'] = df['realized_vol'].shift(lag)
        
        # Market regime indicators
        df['vol_regime'] = (df['realized_vol'] > df['realized_vol'].rolling(252).quantile(0.75)).astype(int)
        df['trend_regime'] = (df['price_sma_20'] > df['price_sma_50']).astype(int)
        
        return df
    
    def _parkinson_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Parkinson volatility estimator."""
        if 'high' not in df.columns or 'low' not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        
        return np.sqrt(np.log(df['high'] / df['low']) ** 2 / (4 * np.log(2))) * np.sqrt(252)
    
    def _garman_klass_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Garman-Klass volatility estimator."""
        if not all(col in df.columns for col in ['high', 'low', 'open', 'close']):
            return pd.Series(index=df.index, dtype=float)
        
        rs = np.log(df['high'] / df['close']) * np.log(df['high'] / df['open'])
        rs += np.log(df['low'] / df['close']) * np.log(df['low'] / df['open'])
        
        return np.sqrt(rs) * np.sqrt(252)
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _bollinger_bands(self, prices: pd.Series, window: int = 20, 
                        num_std: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, lower
    
    def prepare_lstm_sequences(self, data: pd.DataFrame, 
                              target_col: str,
                              feature_cols: List[str],
                              sequence_length: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        target_col : str
            Target column name
        feature_cols : List[str]
            Feature column names
        sequence_length : int
            Length of input sequences
            
        Returns:
        --------
        Tuple of (X, y) arrays for LSTM training
        """
        # Remove rows with NaN values
        clean_data = data[feature_cols + [target_col]].dropna()
        
        X, y = [], []
        for i in range(sequence_length, len(clean_data)):
            X.append(clean_data[feature_cols].iloc[i-sequence_length:i].values)
            y.append(clean_data[target_col].iloc[i])
        
        return np.array(X), np.array(y)
    
    def create_target_variable(self, data: pd.DataFrame, 
                              method: str = 'statistical',
                              threshold: float = 0.95) -> pd.Series:
        """
        Create binary target variable for extreme volatility events.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with returns
        method : str
            Method for defining extreme events
        threshold : float
            Threshold for extreme events
            
        Returns:
        --------
        Binary series indicating extreme events
        """
        detector = ExtremeVolatilityDetector(threshold_percentile=threshold*100)
        return detector.detect_extreme_events(data['returns'], method=method)


class ModelImplementationExamples:
    """
    Concrete examples of model implementations for thesis.
    """
    
    def __init__(self):
        """Initialize model examples."""
        self.models = {}
        
    def simple_garch_example(self, returns: pd.Series) -> Dict:
        """
        Simple GARCH(1,1) implementation example.
        
        Parameters:
        -----------
        returns : pd.Series
            Time series of returns
            
        Returns:
        --------
        Dict with model results and code example
        """
        example_code = """
# GARCH(1,1) Implementation Example
from arch import arch_model

def fit_garch_model(returns):
    # Remove any NaN values
    clean_returns = returns.dropna() * 100  # Convert to percentage
    
    # Define GARCH(1,1) model
    model = arch_model(clean_returns, vol='Garch', p=1, q=1, dist='normal')
    
    # Fit the model
    fitted_model = model.fit(disp='off')
    
    # Extract conditional volatility
    conditional_vol = fitted_model.conditional_volatility / 100
    
    return fitted_model, conditional_vol

# Alternative distributions
models = {
    'normal': arch_model(returns, vol='Garch', p=1, q=1, dist='normal'),
    't': arch_model(returns, vol='Garch', p=1, q=1, dist='t'),
    'skewt': arch_model(returns, vol='Garch', p=1, q=1, dist='skewt')
}

# Fit all models and compare AIC/BIC
results = {}
for name, model in models.items():
    fitted = model.fit(disp='off')
    results[name] = {
        'aic': fitted.aic,
        'bic': fitted.bic,
        'loglik': fitted.loglikelihood
    }
        """
        
        return {
            'code_example': example_code,
            'description': 'GARCH(1,1) with multiple distributions',
            'key_considerations': [
                'Scale returns appropriately (often multiply by 100)',
                'Compare different distributions (normal, t, skewed-t)',
                'Use information criteria for model selection',
                'Check residual diagnostics'
            ]
        }
    
    def lstm_architecture_example(self) -> Dict:
        """
        LSTM architecture examples for different scenarios.
        
        Returns:
        --------
        Dict with LSTM implementation examples
        """
        example_code = """
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_basic_lstm(sequence_length, n_features):
    \"\"\"Basic LSTM for volatility prediction.\"\"\"
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # For binary classification
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def create_advanced_lstm_with_attention(sequence_length, n_features):
    \"\"\"LSTM with attention mechanism for sentiment integration.\"\"\"
    from tensorflow.keras.layers import Attention, Concatenate
    
    # Price input branch
    price_input = tf.keras.Input(shape=(sequence_length, n_features-1))
    price_lstm = LSTM(64, return_sequences=True)(price_input)
    price_lstm = Dropout(0.2)(price_lstm)
    
    # Sentiment input branch
    sentiment_input = tf.keras.Input(shape=(sequence_length, 1))
    sentiment_lstm = LSTM(32, return_sequences=True)(sentiment_input)
    
    # Attention mechanism
    attention = Attention()([price_lstm, sentiment_lstm])
    
    # Combine features
    combined = Concatenate()([price_lstm, attention])
    combined = LSTM(32, return_sequences=False)(combined)
    combined = Dropout(0.2)(combined)
    
    # Output layers
    output = Dense(16, activation='relu')(combined)
    output = Dense(1, activation='sigmoid')(output)
    
    model = tf.keras.Model(inputs=[price_input, sentiment_input], outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

# Training with callbacks
def train_lstm_with_callbacks(model, X_train, y_train, X_val, y_val):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return history
        """
        
        return {
            'code_example': example_code,
            'architectures': {
                'basic_lstm': 'Simple 2-layer LSTM for baseline comparison',
                'attention_lstm': 'LSTM with attention for sentiment integration',
                'ensemble_lstm': 'Multiple LSTM models with different configurations'
            },
            'hyperparameter_ranges': {
                'lstm_units': [32, 64, 128],
                'dropout_rate': [0.1, 0.2, 0.3],
                'learning_rate': [1e-4, 1e-3, 1e-2],
                'sequence_length': [10, 20, 50],
                'batch_size': [16, 32, 64]
            }
        }
    
    def evaluation_framework_example(self) -> Dict:
        """
        Comprehensive evaluation framework example.
        
        Returns:
        --------
        Dict with evaluation code examples
        """
        example_code = """
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix)
import matplotlib.pyplot as plt

def comprehensive_model_evaluation(y_true, y_pred, y_pred_proba, returns=None):
    \"\"\"Comprehensive evaluation of extreme volatility prediction models.\"\"\"
    
    # Statistical metrics
    statistical_metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
    }
    
    # Financial metrics (if returns provided)
    financial_metrics = {}
    if returns is not None:
        strategy_returns = returns * y_pred  # Simple strategy
        financial_metrics = {
            'sharpe_ratio': calculate_sharpe_ratio(strategy_returns),
            'max_drawdown': calculate_max_drawdown(strategy_returns),
            'calmar_ratio': calculate_calmar_ratio(strategy_returns),
            'hit_rate': calculate_hit_rate(y_true, y_pred)
        }
    
    # Risk metrics
    risk_metrics = calculate_risk_metrics(y_true, y_pred, returns)
    
    return {
        'statistical': statistical_metrics,
        'financial': financial_metrics,
        'risk': risk_metrics
    }

def model_comparison_tests(model_predictions, y_true, returns):
    \"\"\"Statistical tests for model comparison.\"\"\"
    from scipy import stats
    
    results = {}
    
    # Diebold-Mariano test for forecast accuracy
    for model1, model2 in combinations(model_predictions.keys(), 2):
        dm_stat, p_value = diebold_mariano_test(
            model_predictions[model1], 
            model_predictions[model2], 
            y_true
        )
        results[f'{model1}_vs_{model2}'] = {
            'dm_statistic': dm_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    return results

def create_evaluation_plots(y_true, y_pred_proba, model_name):
    \"\"\"Create comprehensive evaluation plots.\"\"\"
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    axes[0, 0].plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_true, y_pred_proba):.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    
    # Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    axes[0, 1].plot(recall, precision)
    axes[0, 1].set_title('Precision-Recall Curve')
    
    # Prediction distribution
    axes[1, 0].hist(y_pred_proba[y_true == 0], alpha=0.5, label='Normal', bins=20)
    axes[1, 0].hist(y_pred_proba[y_true == 1], alpha=0.5, label='Extreme', bins=20)
    axes[1, 0].set_title('Prediction Distributions')
    axes[1, 0].legend()
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred_proba > 0.5)
    axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 1].set_title('Confusion Matrix')
    
    plt.tight_layout()
    return fig
        """
        
        return {
            'code_example': example_code,
            'evaluation_dimensions': [
                'Statistical performance (accuracy, precision, recall)',
                'Financial performance (Sharpe ratio, drawdown)',
                'Risk metrics (VaR, Expected Shortfall)',
                'Stability across time periods',
                'Statistical significance tests'
            ],
            'visualization_types': [
                'ROC and Precision-Recall curves',
                'Prediction distribution plots',
                'Time series of predictions vs actuals',
                'Performance attribution analysis'
            ]
        }


class ThesisWritingGuide:
    """
    Guide for writing different sections of the thesis.
    """
    
    def __init__(self):
        """Initialize writing guide."""
        pass
    
    def literature_review_structure(self) -> Dict[str, List[str]]:
        """
        Provide structure for literature review section.
        
        Returns:
        --------
        Dict with literature review structure
        """
        structure = {
            'volatility_modeling_foundations': [
                'ARCH/GARCH model development (Engle 1982, Bollerslev 1986)',
                'Extensions: EGARCH, GJR-GARCH, FIGARCH',
                'Stylized facts of financial volatility',
                'Model specification and diagnostic testing'
            ],
            'machine_learning_in_finance': [
                'Neural networks for financial prediction (White 1988)',
                'Deep learning applications in finance (Heaton et al. 2017)',
                'LSTM for time series (Hochreiter & Schmidhuber 1997)',
                'Overfitting and generalization challenges'
            ],
            'sentiment_analysis_integration': [
                'Behavioral finance foundations (Kahneman & Tversky)',
                'Text mining for financial prediction (Tetlock 2007)',
                'Social media sentiment and markets (Bollen et al. 2011)',
                'Multi-modal fusion techniques'
            ],
            'comparative_studies': [
                'Traditional vs ML models in finance',
                'Ensemble methods and model combination',
                'Performance evaluation frameworks',
                'Economic significance vs statistical significance'
            ]
        }
        return structure
    
    def methodology_chapter_outline(self) -> Dict[str, List[str]]:
        """
        Provide outline for methodology chapter.
        
        Returns:
        --------
        Dict with methodology chapter structure
        """
        outline = {
            'data_description': [
                'WTI crude oil futures data sources and characteristics',
                'Sample period justification and market regime coverage',
                'Data cleaning and preprocessing procedures',
                'Descriptive statistics and stylized facts'
            ],
            'extreme_events_definition': [
                'Multiple operational definitions of extreme volatility',
                'Threshold selection methodology and sensitivity analysis',
                'Event frequency and clustering analysis',
                'Comparison with literature definitions'
            ],
            'model_specifications': [
                'GARCH family models: specification and estimation',
                'LSTM architecture design and hyperparameter selection',
                'Sentiment data integration methodology',
                'Ensemble and hybrid model approaches'
            ],
            'evaluation_framework': [
                'Train/validation/test split methodology',
                'Statistical performance metrics',
                'Financial performance evaluation',
                'Model comparison statistical tests',
                'Robustness checks and sensitivity analysis'
            ]
        }
        return outline
    
    def results_presentation_guide(self) -> Dict[str, List[str]]:
        """
        Guide for presenting results effectively.
        
        Returns:
        --------
        Dict with results presentation guidelines
        """
        guide = {
            'descriptive_results': [
                'Summary statistics table for all variables',
                'Extreme events timeline and frequency analysis',
                'Correlation analysis between variables',
                'Regime identification and characterization'
            ],
            'model_performance_tables': [
                'Statistical metrics comparison (precision, recall, F1)',
                'Financial performance comparison (Sharpe, Calmar)',
                'Model stability across different periods',
                'Statistical significance tests results'
            ],
            'visualization_best_practices': [
                'Time series plots with event highlighting',
                'ROC curves for all models on same plot',
                'Rolling window performance analysis',
                'Feature importance and attribution analysis'
            ],
            'robustness_analysis': [
                'Sensitivity to hyperparameters',
                'Performance across different market regimes',
                'Out-of-sample period analysis',
                'Bootstrap confidence intervals'
            ]
        }
        return guide


def create_thesis_template_structure():
    """
    Create a complete thesis template structure.
    
    Returns:
    --------
    Dict with complete thesis structure and content guidelines
    """
    template = {
        'title_suggestions': [
            'Extreme Volatility Prediction in WTI Crude Oil: A Comparative Analysis of GARCH, LSTM, and Sentiment-Enhanced Models',
            'Deep Learning vs Traditional Econometric Models for Extreme Volatility Events: Evidence from WTI Crude Oil Markets',
            'Integrating Sentiment Analysis with Deep Learning for Extreme Volatility Prediction in Energy Markets'
        ],
        
        'abstract_structure': {
            'background': 'Extreme volatility events in commodity markets pose significant challenges...',
            'objective': 'This study compares the performance of traditional GARCH models...',
            'methodology': 'Using WTI crude oil data from 2015-2024, we implement...',
            'results': 'LSTM models show superior performance in capturing...',
            'conclusion': 'The integration of sentiment analysis provides...',
            'implications': 'These findings have important implications for risk management...'
        },
        
        'chapter_structure': {
            'chapter_1_introduction': [
                'Research motivation and background',
                'Problem statement and research questions',
                'Research objectives and hypotheses',
                'Contribution and significance',
                'Thesis structure overview'
            ],
            'chapter_2_literature_review': [
                'Volatility modeling in financial markets',
                'Machine learning applications in finance',
                'Sentiment analysis and behavioral finance',
                'Comparative studies and gaps in literature'
            ],
            'chapter_3_methodology': [
                'Data description and preprocessing',
                'Extreme events definition and detection',
                'Model specifications and architectures',
                'Evaluation framework and metrics'
            ],
            'chapter_4_results': [
                'Descriptive analysis and data characteristics',
                'Model performance comparison',
                'Statistical significance testing',
                'Robustness analysis'
            ],
            'chapter_5_discussion': [
                'Interpretation of results',
                'Comparison with existing literature',
                'Practical implications',
                'Limitations and future research'
            ],
            'chapter_6_conclusion': [
                'Summary of findings',
                'Contributions to knowledge',
                'Practical applications',
                'Future research directions'
            ]
        },
        
        'appendices': [
            'Additional statistical tests',
            'Hyperparameter optimization details',
            'Code implementation details',
            'Additional robustness checks'
        ]
    }
    
    return template


if __name__ == "__main__":
    print("=== Thesis Methodology Examples Demonstration ===\n")
    
    # 1. Data preprocessing example
    print("1. DATA PREPROCESSING EXAMPLE")
    print("-" * 40)
    
    # Generate sample data
    financial_data, sentiment_data = generate_sample_data()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Add some synthetic OHLC data for demonstration
    financial_data['high'] = financial_data['price'] * (1 + np.random.uniform(0, 0.02, len(financial_data)))
    financial_data['low'] = financial_data['price'] * (1 - np.random.uniform(0, 0.02, len(financial_data)))
    financial_data['open'] = financial_data['price'] * (1 + np.random.uniform(-0.01, 0.01, len(financial_data)))
    financial_data['close'] = financial_data['price']
    
    # Create features
    features_df = preprocessor.create_financial_features(financial_data)
    print(f"Created {len(features_df.columns)} features from price data")
    print(f"Feature columns: {list(features_df.columns[:10])}...")  # Show first 10
    
    # Create target variable
    target = preprocessor.create_target_variable(features_df)
    print(f"Target variable: {target.sum()} extreme events out of {len(target)} observations")
    
    # 2. Model implementation examples
    print("\n\n2. MODEL IMPLEMENTATION EXAMPLES")
    print("-" * 40)
    
    model_examples = ModelImplementationExamples()
    
    # GARCH example
    garch_example = model_examples.simple_garch_example(financial_data['returns'])
    print("GARCH Implementation Guidelines:")
    for consideration in garch_example['key_considerations']:
        print(f"  • {consideration}")
    
    # LSTM example
    lstm_example = model_examples.lstm_architecture_example()
    print(f"\nLSTM Architectures Available:")
    for arch, desc in lstm_example['architectures'].items():
        print(f"  • {arch}: {desc}")
    
    # 3. Thesis writing guide
    print("\n\n3. THESIS WRITING STRUCTURE")
    print("-" * 40)
    
    writing_guide = ThesisWritingGuide()
    
    # Literature review structure
    lit_structure = writing_guide.literature_review_structure()
    print("Literature Review Structure:")
    for section, items in lit_structure.items():
        print(f"\n{section.upper().replace('_', ' ')}:")
        for item in items[:2]:  # Show first 2 items
            print(f"  • {item}")
    
    # 4. Complete thesis template
    print("\n\n4. COMPLETE THESIS TEMPLATE")
    print("-" * 40)
    
    template = create_thesis_template_structure()
    print("Suggested Thesis Titles:")
    for i, title in enumerate(template['title_suggestions'], 1):
        print(f"  {i}. {title}")
    
    print(f"\nChapter Structure: {len(template['chapter_structure'])} chapters")
    for chapter, sections in template['chapter_structure'].items():
        print(f"  • {chapter.replace('_', ' ').title()}: {len(sections)} sections")
    
    print("\n=== Methodology examples successfully demonstrated! ===")
    print("\nThese examples provide concrete implementation guidance")
    print("for developing a robust financial time series thesis.")