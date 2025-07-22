"""
Financial Time Series Analysis for Thesis Development
=====================================================

This module provides practical implementations and frameworks for developing
a thesis on extreme volatility events in WTI crude oil using GARCH, LSTM,
and sentiment analysis models.

Academic Context: Data Science & Business Analytics Thesis Guidance
Focus: Structural and strategic guidance for robust financial time series analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports (when available)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("TensorFlow not available. LSTM implementations will be limited.")

# Financial modeling imports
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    print("ARCH package not available. GARCH implementations will be limited.")

# Sentiment analysis imports
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    print("VADER Sentiment not available. Sentiment analysis will be limited.")


class ThesisFramework:
    """
    Comprehensive framework for financial time series thesis development.
    
    This class provides academic guidance and practical tools for developing
    a thesis on extreme volatility events using multiple modeling approaches.
    """
    
    def __init__(self, data_frequency: str = 'daily'):
        """
        Initialize the thesis framework.
        
        Parameters:
        -----------
        data_frequency : str
            Frequency of the data ('daily', 'intraday', 'weekly')
        """
        self.data_frequency = data_frequency
        self.models = {}
        self.results = {}
        self.evaluation_metrics = {}
        
    def evaluate_thesis_viability(self) -> Dict[str, Dict[str, str]]:
        """
        Evaluate the viability, defensibility, and productivity of the thesis approach.
        
        Returns:
        --------
        Dict containing detailed evaluation of the thesis approach
        """
        evaluation = {
            'viability': {
                'technical_feasibility': 'HIGH - GARCH and LSTM are well-established methods',
                'data_availability': 'HIGH - WTI has extensive historical data',
                'computational_requirements': 'MEDIUM - Standard deep learning setup sufficient',
                'timeline_feasibility': 'HIGH - 4-6 months for complete implementation'
            },
            'defensibility': {
                'theoretical_foundation': 'STRONG - Solid econometric and ML literature base',
                'methodological_rigor': 'HIGH - Clear comparison framework possible',
                'novelty': 'MEDIUM-HIGH - Systematic comparison with sentiment integration',
                'practical_relevance': 'HIGH - Direct applications in risk management'
            },
            'productivity': {
                'academic_contribution': 'SIGNIFICANT - Bridges econometrics and ML',
                'industry_impact': 'HIGH - Risk management and trading applications',
                'reproducibility': 'HIGH - With proper implementation',
                'extensibility': 'HIGH - Framework applicable to other commodities'
            }
        }
        return evaluation
    
    def design_methodology(self) -> Dict[str, List[str]]:
        """
        Provide structured methodology design for the thesis.
        
        Returns:
        --------
        Dict containing detailed methodology framework
        """
        methodology = {
            'data_architecture': [
                'WTI Futures daily/intraday prices (2015-2024)',
                'Macroeconomic variables (USD Index, VIX, Treasury yields)',
                'News sentiment data (financial news, social media)',
                'Oil market fundamentals (EIA inventories, OPEC production)',
                'Geopolitical risk indicators'
            ],
            'extreme_events_definition': [
                'Statistical: Realized volatility > 95th percentile',
                'Economic: Daily price changes > 5%',
                'Persistence: High volatility lasting 3+ consecutive days',
                'Contextual: Events during known crisis periods'
            ],
            'model_comparison_framework': [
                'GARCH(1,1), eGARCH, GJR-GARCH with different distributions',
                'LSTM with various architectures (2-3 layers, 50-100 units)',
                'LSTM + Sentiment with attention mechanisms',
                'Ensemble methods combining multiple approaches'
            ],
            'validation_strategy': [
                'Walk-forward analysis (no look-ahead bias)',
                'Multiple regime testing (crisis vs normal periods)',
                'Cross-validation with temporal constraints',
                'Out-of-sample testing on recent data'
            ]
        }
        return methodology
    
    def common_thesis_pitfalls(self) -> Dict[str, Dict[str, str]]:
        """
        Identify and provide solutions for common thesis errors.
        
        Returns:
        --------
        Dict containing common errors and their solutions
        """
        pitfalls = {
            'data_leakage_issues': {
                'look_ahead_bias': 'Use strict temporal splitting, never train on future data',
                'sentiment_leakage': 'Ensure sentiment data timestamps precede price movements',
                'preprocessing_leakage': 'Apply scaling/normalization only on training data'
            },
            'methodological_errors': {
                'overfitting': 'Use dropout, early stopping, cross-validation with time constraints',
                'survivorship_bias': 'Include delisted contracts, account for contract rollovers',
                'multiple_testing': 'Apply Bonferroni or FDR corrections for multiple comparisons'
            },
            'evaluation_mistakes': {
                'wrong_metrics': 'Use financial metrics (Sharpe, Max DD) not just statistical ones',
                'inadequate_testing': 'Test across multiple market regimes, not just overall period',
                'significance_issues': 'Bootstrap confidence intervals, Diebold-Mariano tests'
            },
            'reproducibility_failures': {
                'random_seeds': 'Set all random seeds for reproducible results',
                'environment_docs': 'Document exact package versions and hardware specs',
                'code_organization': 'Use clear project structure with version control'
            }
        }
        return pitfalls
    
    def ensure_dl_effectiveness(self) -> Dict[str, List[str]]:
        """
        Guidelines to ensure Deep Learning models are effective and replicable.
        
        Returns:
        --------
        Dict containing effectiveness and replicability guidelines
        """
        guidelines = {
            'architecture_design': [
                'Start simple, increase complexity gradually',
                'Use regularization: dropout (0.2-0.3), L1/L2 penalties',
                'Implement gradient clipping to prevent exploding gradients',
                'Apply batch normalization for training stability'
            ],
            'training_strategy': [
                'Learning rate scheduling (warm-up + decay)',
                'Early stopping with patience (10-15 epochs)',
                'Cross-validation with temporal constraints',
                'Ensemble multiple models for robustness'
            ],
            'hyperparameter_optimization': [
                'Systematic grid/random search over reasonable ranges',
                'Use validation set for hyperparameter selection',
                'Document all hyperparameter choices and rationale',
                'Test sensitivity to key hyperparameters'
            ],
            'replicability_measures': [
                'Set random seeds for all operations',
                'Document exact software versions',
                'Provide complete data preprocessing pipeline',
                'Include model checkpoints and training logs'
            ]
        }
        return guidelines
    
    def temporal_analysis_impact(self) -> Dict[str, str]:
        """
        Analyze how study time period affects model capability for non-linear relationships.
        
        Returns:
        --------
        Dict containing analysis of temporal impact on model performance
        """
        impact_analysis = {
            'minimum_period': '5 years minimum to capture at least 1-2 complete cycles',
            'optimal_period': '10-15 years to include multiple crisis and normal periods',
            'maximum_period': '20+ years risks structural breaks, requires stability testing',
            'regime_considerations': 'Different volatility regimes require separate analysis',
            'rolling_windows': 'Use 1-3 year rolling windows for model stability testing',
            'structural_breaks': 'Test for structural breaks using Chow tests, CUSUM statistics',
            'adaptation_strategy': 'Implement online learning for model parameter updates',
            'sequence_length': 'LSTM sequence length should match the dominant cycle length'
        }
        return impact_analysis


class ExtremeVolatilityDetector:
    """
    Framework for detecting and analyzing extreme volatility events.
    """
    
    def __init__(self, threshold_percentile: float = 95):
        """
        Initialize extreme volatility detector.
        
        Parameters:
        -----------
        threshold_percentile : float
            Percentile threshold for extreme events (default: 95)
        """
        self.threshold_percentile = threshold_percentile
        self.extreme_events = None
        
    def detect_extreme_events(self, returns: pd.Series, 
                            method: str = 'statistical') -> pd.Series:
        """
        Detect extreme volatility events using specified method.
        
        Parameters:
        -----------
        returns : pd.Series
            Time series of returns
        method : str
            Detection method ('statistical', 'economic', 'persistence')
            
        Returns:
        --------
        pd.Series of boolean indicators for extreme events
        """
        if method == 'statistical':
            # Statistical threshold based on percentile
            volatility = returns.rolling(window=20).std() * np.sqrt(252)
            threshold = np.percentile(volatility.dropna(), self.threshold_percentile)
            extreme_events = volatility > threshold
            
        elif method == 'economic':
            # Economic threshold based on absolute returns
            threshold = 0.05  # 5% daily move
            extreme_events = np.abs(returns) > threshold
            
        elif method == 'persistence':
            # Persistence-based detection
            volatility = returns.rolling(window=5).std()
            high_vol = volatility > volatility.quantile(0.9)
            extreme_events = high_vol.rolling(window=3).sum() >= 3
            
        else:
            raise ValueError(f"Unknown method: {method}")
            
        self.extreme_events = extreme_events.fillna(False)
        return self.extreme_events
    
    def analyze_event_characteristics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Analyze characteristics of detected extreme events.
        
        Parameters:
        -----------
        returns : pd.Series
            Time series of returns
            
        Returns:
        --------
        Dict containing event characteristics
        """
        if self.extreme_events is None:
            raise ValueError("Must detect extreme events first")
            
        extreme_returns = returns[self.extreme_events]
        
        characteristics = {
            'frequency': self.extreme_events.sum() / len(self.extreme_events),
            'average_magnitude': np.abs(extreme_returns).mean(),
            'skewness': extreme_returns.skew(),
            'kurtosis': extreme_returns.kurtosis(),
            'clustering_coefficient': self._calculate_clustering(),
            'average_duration': self._calculate_average_duration()
        }
        
        return characteristics
    
    def _calculate_clustering(self) -> float:
        """Calculate clustering coefficient of extreme events."""
        if self.extreme_events is None:
            return 0.0
        
        # Simple clustering measure: consecutive extreme events
        consecutive = 0
        total_events = self.extreme_events.sum()
        
        if total_events == 0:
            return 0.0
            
        for i in range(1, len(self.extreme_events)):
            if self.extreme_events.iloc[i] and self.extreme_events.iloc[i-1]:
                consecutive += 1
                
        return consecutive / total_events if total_events > 0 else 0.0
    
    def _calculate_average_duration(self) -> float:
        """Calculate average duration of extreme event periods."""
        if self.extreme_events is None:
            return 0.0
            
        durations = []
        current_duration = 0
        
        for event in self.extreme_events:
            if event:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
                    
        # Handle case where series ends with extreme event
        if current_duration > 0:
            durations.append(current_duration)
            
        return np.mean(durations) if durations else 0.0


class ModelComparison:
    """
    Framework for comparing GARCH, LSTM, and LSTM+Sentiment models.
    """
    
    def __init__(self):
        """Initialize model comparison framework."""
        self.models = {}
        self.results = {}
        
    def create_garch_model(self, returns: pd.Series, 
                          model_type: str = 'GARCH') -> 'arch.univariate.ARCHModel':
        """
        Create and fit GARCH model.
        
        Parameters:
        -----------
        returns : pd.Series
            Time series of returns
        model_type : str
            Type of GARCH model ('GARCH', 'EGARCH', 'GJR-GARCH')
            
        Returns:
        --------
        Fitted ARCH model
        """
        if not HAS_ARCH:
            raise ImportError("ARCH package required for GARCH models")
            
        if model_type == 'GARCH':
            model = arch_model(returns, vol='Garch', p=1, q=1)
        elif model_type == 'EGARCH':
            model = arch_model(returns, vol='EGARCH', p=1, q=1)
        elif model_type == 'GJR-GARCH':
            model = arch_model(returns, vol='GJRGARCH', p=1, q=1)
        else:
            raise ValueError(f"Unknown GARCH model type: {model_type}")
            
        fitted_model = model.fit(disp='off')
        self.models[f'garch_{model_type.lower()}'] = fitted_model
        
        return fitted_model
    
    def create_lstm_model(self, sequence_length: int = 20, 
                         features: int = 1,
                         include_sentiment: bool = False) -> 'tf.keras.Model':
        """
        Create LSTM model architecture.
        
        Parameters:
        -----------
        sequence_length : int
            Length of input sequences
        features : int
            Number of input features
        include_sentiment : bool
            Whether to include sentiment features
            
        Returns:
        --------
        Compiled Keras model
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow required for LSTM models")
            
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(64, return_sequences=True, 
                      input_shape=(sequence_length, features)))
        model.add(Dropout(0.2))
        
        # Second LSTM layer
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Dense layers
        if include_sentiment:
            model.add(Dense(16, activation='relu'))
            model.add(Dropout(0.1))
            
        model.add(Dense(1, activation='sigmoid'))  # For binary classification
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy', 'precision', 'recall'])
        
        model_name = 'lstm_sentiment' if include_sentiment else 'lstm_basic'
        self.models[model_name] = model
        
        return model
    
    def evaluate_model_performance(self, model_name: str, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate model performance using multiple metrics.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_pred_proba : np.ndarray, optional
            Predicted probabilities
            
        Returns:
        --------
        Dict containing performance metrics
        """
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                   f1_score, roc_auc_score, average_precision_score)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_pred_proba is not None:
            metrics.update({
                'roc_auc': roc_auc_score(y_true, y_pred_proba),
                'avg_precision': average_precision_score(y_true, y_pred_proba)
            })
            
        self.results[model_name] = metrics
        return metrics
    
    def financial_performance_metrics(self, returns: pd.Series, 
                                    predictions: pd.Series) -> Dict[str, float]:
        """
        Calculate financial performance metrics.
        
        Parameters:
        -----------
        returns : pd.Series
            Time series of returns
        predictions : pd.Series
            Model predictions for trading signals
            
        Returns:
        --------
        Dict containing financial metrics
        """
        # Simple strategy: go long when model predicts extreme volatility
        strategy_returns = returns * predictions.shift(1)
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) == 0:
            return {'error': 'No valid strategy returns'}
        
        # Calculate metrics
        annual_return = strategy_returns.mean() * 252
        annual_volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # VaR and Expected Shortfall
        var_95 = np.percentile(strategy_returns, 5)
        expected_shortfall = strategy_returns[strategy_returns <= var_95].mean()
        
        financial_metrics = {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'expected_shortfall': expected_shortfall,
            'calmar_ratio': annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        }
        
        return financial_metrics


class SentimentAnalyzer:
    """
    Framework for analyzing sentiment data for financial time series.
    """
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        if HAS_VADER:
            self.analyzer = SentimentIntensityAnalyzer()
        else:
            self.analyzer = None
            
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using VADER.
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        Dict containing sentiment scores
        """
        if not self.analyzer:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
            
        scores = self.analyzer.polarity_scores(text)
        return scores
    
    def create_sentiment_features(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create sentiment features from news data.
        
        Parameters:
        -----------
        news_data : pd.DataFrame
            DataFrame with 'date' and 'text' columns
            
        Returns:
        --------
        DataFrame with sentiment features
        """
        if not self.analyzer:
            # Return dummy features if VADER not available
            dates = pd.to_datetime(news_data['date'])
            return pd.DataFrame({
                'date': dates,
                'sentiment_score': np.random.normal(0, 0.1, len(dates)),
                'sentiment_volatility': np.random.uniform(0, 0.2, len(dates)),
                'news_volume': np.random.poisson(10, len(dates))
            })
        
        sentiments = []
        for _, row in news_data.iterrows():
            sentiment = self.analyze_text_sentiment(row['text'])
            sentiment['date'] = row['date']
            sentiments.append(sentiment)
            
        sentiment_df = pd.DataFrame(sentiments)
        
        # Aggregate by date
        daily_sentiment = sentiment_df.groupby('date').agg({
            'compound': ['mean', 'std', 'count'],
            'pos': 'mean',
            'neg': 'mean'
        }).round(4)
        
        # Flatten column names
        daily_sentiment.columns = [
            'sentiment_score', 'sentiment_volatility', 'news_volume',
            'positive_sentiment', 'negative_sentiment'
        ]
        
        daily_sentiment['sentiment_volatility'] = daily_sentiment['sentiment_volatility'].fillna(0)
        
        return daily_sentiment.reset_index()


def generate_sample_data(start_date: str = '2020-01-01', 
                        end_date: str = '2024-01-01',
                        include_sentiment: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate sample financial and sentiment data for testing.
    
    Parameters:
    -----------
    start_date : str
        Start date for data generation
    end_date : str
        End date for data generation
    include_sentiment : bool
        Whether to include sentiment data
        
    Returns:
    --------
    Tuple of (financial_data, sentiment_data) DataFrames
    """
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate synthetic WTI price data with volatility clustering
    np.random.seed(42)
    n_days = len(dates)
    
    # GARCH-like volatility process
    volatility = np.zeros(n_days)
    volatility[0] = 0.02
    
    for i in range(1, n_days):
        volatility[i] = 0.00001 + 0.05 * (0.01 ** 2) + 0.9 * volatility[i-1]
        
    # Generate returns with time-varying volatility
    returns = np.random.normal(0, 1, n_days) * np.sqrt(volatility)
    
    # Generate prices
    prices = np.zeros(n_days)
    prices[0] = 70.0  # Starting WTI price
    
    for i in range(1, n_days):
        prices[i] = prices[i-1] * (1 + returns[i])
    
    # Create financial dataframe
    financial_data = pd.DataFrame({
        'date': dates,
        'price': prices,
        'returns': returns,
        'volume': np.random.lognormal(15, 0.3, n_days),
        'vix': np.random.normal(20, 5, n_days),
        'usd_index': np.random.normal(100, 2, n_days)
    })
    
    # Generate sentiment data if requested
    sentiment_data = None
    if include_sentiment:
        # Sample news headlines (placeholder)
        news_texts = [
            "Oil prices surge amid geopolitical tensions",
            "Crude futures decline on supply concerns",
            "Energy markets show volatility",
            "OPEC production cuts boost oil sentiment",
            "Refiners report strong demand"
        ] * (n_days // 5 + 1)
        
        sentiment_analyzer = SentimentAnalyzer()
        news_data = pd.DataFrame({
            'date': dates[:len(news_texts)],
            'text': news_texts[:len(dates)]
        })
        
        sentiment_data = sentiment_analyzer.create_sentiment_features(news_data)
    
    return financial_data, sentiment_data


# Example usage and demonstration
def demonstrate_thesis_framework():
    """
    Demonstrate the thesis framework with practical examples.
    """
    print("=== Financial Time Series Thesis Framework Demonstration ===\n")
    
    # Initialize framework
    framework = ThesisFramework()
    
    # 1. Evaluate thesis viability
    print("1. THESIS VIABILITY EVALUATION")
    print("-" * 40)
    viability = framework.evaluate_thesis_viability()
    for category, aspects in viability.items():
        print(f"\n{category.upper()}:")
        for aspect, evaluation in aspects.items():
            print(f"  • {aspect}: {evaluation}")
    
    # 2. Show methodology design
    print("\n\n2. METHODOLOGY DESIGN")
    print("-" * 40)
    methodology = framework.design_methodology()
    for category, items in methodology.items():
        print(f"\n{category.upper()}:")
        for item in items:
            print(f"  • {item}")
    
    # 3. Common pitfalls
    print("\n\n3. COMMON THESIS PITFALLS TO AVOID")
    print("-" * 40)
    pitfalls = framework.common_thesis_pitfalls()
    for category, issues in pitfalls.items():
        print(f"\n{category.upper()}:")
        for issue, solution in issues.items():
            print(f"  • {issue}: {solution}")
    
    # 4. Generate sample data and demonstrate analysis
    print("\n\n4. PRACTICAL DEMONSTRATION WITH SAMPLE DATA")
    print("-" * 40)
    
    # Generate sample data
    financial_data, sentiment_data = generate_sample_data()
    print(f"Generated sample data: {len(financial_data)} days")
    print(f"Financial data columns: {list(financial_data.columns)}")
    if sentiment_data is not None:
        print(f"Sentiment data columns: {list(sentiment_data.columns)}")
    
    # Demonstrate extreme event detection
    detector = ExtremeVolatilityDetector(threshold_percentile=95)
    extreme_events = detector.detect_extreme_events(financial_data['returns'])
    print(f"\nDetected {extreme_events.sum()} extreme volatility events ({extreme_events.mean():.2%} of days)")
    
    # Analyze event characteristics
    characteristics = detector.analyze_event_characteristics(financial_data['returns'])
    print("\nExtreme Event Characteristics:")
    for char, value in characteristics.items():
        print(f"  • {char}: {value:.4f}")
    
    print("\n=== Framework successfully demonstrated! ===")
    print("\nThis framework provides comprehensive guidance for developing")
    print("a robust thesis on financial time series analysis using")
    print("GARCH, LSTM, and sentiment analysis models.")


if __name__ == "__main__":
    demonstrate_thesis_framework()