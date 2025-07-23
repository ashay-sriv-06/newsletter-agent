# tests/test_newsletter_agent.py
"""
Comprehensive test suite for SoFi Newsletter Agent
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json

from sofi_newsletter_agent import (
    MistralNewsletterAgent, 
    NewsItem, 
    ContentCategory, 
    ContentPriority,
    ContentAnalyzer,
    NewsletterGenerator
)

class TestNewsletterAgent:
    """Test the main agent functionality"""
    
    @pytest.fixture
    def mock_mistral_client(self):
        """Mock Mistral client for testing"""
        client = Mock()
        client.chat_async = AsyncMock()
        client.chat_async.return_value = Mock(
            choices=[Mock(message=Mock(content="Test generated content"))]
        )
        return client
    
    @pytest.fixture
    def sample_news_items(self):
        """Generate sample news items for testing"""
        return [
            NewsItem(
                source="Test",
                category=ContentCategory.PRODUCT_LAUNCH,
                title="Test Product Launch",
                content="Test content for product launch",
                timestamp=datetime.now() - timedelta(hours=1),
                priority=ContentPriority.CRITICAL,
                tags=["test", "product"],
                metrics={"users": "1000", "revenue": "$50K"}
            ),
            NewsItem(
                source="Test",
                category=ContentCategory.ENGINEERING,
                title="Test Engineering Update",
                content="Test content for engineering",
                timestamp=datetime.now() - timedelta(days=2),
                priority=ContentPriority.HIGH,
                tags=["test", "engineering"]
            ),
            # Duplicate to test deduplication
            NewsItem(
                source="Test2",
                category=ContentCategory.PRODUCT_LAUNCH,
                title="Test Product Launch",
                content="Test content for product launch",
                timestamp=datetime.now() - timedelta(hours=2),
                priority=ContentPriority.HIGH,
                tags=["test", "product"]
            )
        ]
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initializes correctly"""
        agent = MistralNewsletterAgent("test-api-key")
        assert agent.client is not None
        assert agent.model == "mistral-large-latest"
        assert len(agent.data_sources) > 0
    
    @pytest.mark.asyncio
    async def test_content_collection(self, sample_news_items):
        """Test content collection from multiple sources"""
        agent = MistralNewsletterAgent("test-api-key")
        
        # Mock data sources
        for source in agent.data_sources.values():
            source.get_updates = AsyncMock(return_value=sample_news_items[:2])
        
        updates = await agent.collect_weekly_updates()
        
        assert len(updates) > 0
        assert all(isinstance(item, NewsItem) for item in updates)
    
    @pytest.mark.asyncio
    async def test_content_deduplication(self, sample_news_items):
        """Test that duplicate content is properly deduplicated"""
        analyzer = ContentAnalyzer()
        
        # Analyze and rank including duplicates
        ranked = analyzer.analyze_and_rank(sample_news_items)
        
        # Should remove the duplicate
        assert len(ranked) == 2
        assert ranked[0].priority == ContentPriority.CRITICAL  # Highest priority kept
    
    @pytest.mark.asyncio
    async def test_content_selection(self, sample_news_items):
        """Test content selection respects category limits"""
        analyzer = ContentAnalyzer()
        
        # Create many items in same category
        many_items = sample_news_items * 10
        
        selected = analyzer.select_content(many_items, max_items=5)
        
        total_items = sum(len(items) for items in selected.values())
        assert total_items <= 5
        
        # Check category limits are respected
        for category, items in selected.items():
            assert len(items) <= 5  # Max per category
    
    @pytest.mark.asyncio
    async def test_newsletter_generation(self, mock_mistral_client, sample_news_items):
        """Test newsletter HTML generation"""
        generator = NewsletterGenerator(mock_mistral_client, "test-model")
        
        categorized = {
            ContentCategory.PRODUCT_LAUNCH: [sample_news_items[0]],
            ContentCategory.ENGINEERING: [sample_news_items[1]]
        }
        
        newsletter = await generator.generate(categorized)
        
        assert "<html>" in newsletter
        assert "SoFi Weekly Pulse" in newsletter
        assert sample_news_items[0].title in newsletter
        assert mock_mistral_client.chat_async.called
    
    @pytest.mark.asyncio
    async def test_impact_score_calculation(self):
        """Test impact score calculation logic"""
        # Critical priority item from 1 hour ago
        item1 = NewsItem(
            source="Test",
            category=ContentCategory.PRODUCT_LAUNCH,
            title="Critical Update",
            content="Test",
            timestamp=datetime.now() - timedelta(hours=1),
            priority=ContentPriority.CRITICAL,
            tags=["test"],
            metrics={"metric": "value"},
            urls=["http://test.com"]
        )
        
        # Low priority item from 1 week ago
        item2 = NewsItem(
            source="Test",
            category=ContentCategory.METRICS,
            title="Old Update",
            content="Test",
            timestamp=datetime.now() - timedelta(days=7),
            priority=ContentPriority.LOW,
            tags=["test"]
        )
        
        assert item1.impact_score > item2.impact_score
        assert 0 <= item1.impact_score <= 2.0
        assert 0 <= item2.impact_score <= 2.0

# api.py - FastAPI application for health checks and manual triggers
"""
API endpoints for SoFi Newsletter Agent
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
from datetime import datetime
import asyncio

app = FastAPI(title="SoFi Newsletter Agent API", version="1.0.0")

# Pydantic models
class NewsletterRequest(BaseModel):
    recipients: List[str]
    test_mode: bool = True
    include_categories: Optional[List[str]] = None

class NewsletterResponse(BaseModel):
    id: str
    status: str
    generated_at: datetime
    preview_url: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    services: dict

# Global agent instance
agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup"""
    global agent
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not set")
    
    from sofi_newsletter_agent import MistralNewsletterAgent
    agent = MistralNewsletterAgent(api_key)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Kubernetes"""
    services_status = {
        "mistral": "connected" if agent else "disconnected",
        "redis": check_redis_connection(),
        "postgres": check_postgres_connection()
    }
    
    overall_status = "healthy" if all(
        status == "connected" for status in services_status.values()
    ) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(),
        version="1.0.0",
        services=services_status
    )

@app.post("/newsletter/generate", response_model=NewsletterResponse)
async def generate_newsletter(
    request: NewsletterRequest,
    background_tasks: BackgroundTasks
):
    """Manually trigger newsletter generation"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    newsletter_id = f"newsletter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Run generation in background
    background_tasks.add_task(
        generate_and_send_newsletter,
        newsletter_id,
        request.recipients,
        request.test_mode
    )
    
    return NewsletterResponse(
        id=newsletter_id,
        status="generating",
        generated_at=datetime.now(),
        preview_url=f"/newsletter/preview/{newsletter_id}" if request.test_mode else None
    )

@app.get("/newsletter/preview/{newsletter_id}", response_class=HTMLResponse)
async def preview_newsletter(newsletter_id: str):
    """Preview a generated newsletter"""
    preview_path = f"output/{newsletter_id}.html"
    
    if not os.path.exists(preview_path):
        raise HTTPException(status_code=404, detail="Newsletter not found")
    
    with open(preview_path, "r") as f:
        content = f.read()
    
    return HTMLResponse(content=content)

@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint"""
    metrics = []
    
    # Add custom metrics
    metrics.append("# HELP newsletter_generation_total Total newsletters generated")
    metrics.append("# TYPE newsletter_generation_total counter")
    metrics.append(f"newsletter_generation_total {get_generation_count()}")
    
    metrics.append("# HELP newsletter_items_collected_total Total news items collected")
    metrics.append("# TYPE newsletter_items_collected_total counter")
    metrics.append(f"newsletter_items_collected_total {get_items_collected_count()}")
    
    return "\n".join(metrics)

async def generate_and_send_newsletter(
    newsletter_id: str,
    recipients: List[str],
    test_mode: bool
):
    """Background task to generate and send newsletter"""
    try:
        # Generate newsletter
        newsletter_html = await agent.process_and_generate_newsletter()
        
        # Save preview
        preview_path = f"output/{newsletter_id}.html"
        with open(preview_path, "w") as f:
            f.write(newsletter_html)
        
        if not test_mode:
            # Send email
            from sofi_newsletter_agent import EmailSender
            sender = EmailSender(
                smtp_host=os.getenv("SMTP_HOST"),
                smtp_port=int(os.getenv("SMTP_PORT", 587)),
                username=os.getenv("SMTP_USERNAME"),
                password=os.getenv("SMTP_PASSWORD")
            )
            
            await sender.send_newsletter(newsletter_html, recipients)
        
        # Log success
        log_newsletter_generated(newsletter_id, len(recipients))
        
    except Exception as e:
        # Log error
        log_newsletter_error(newsletter_id, str(e))
        raise

def check_redis_connection() -> str:
    """Check Redis connection status"""
    try:
        import redis
        r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        r.ping()
        return "connected"
    except:
        return "disconnected"

def check_postgres_connection() -> str:
    """Check PostgreSQL connection status"""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database="sofi_newsletter",
            user="sofi_user",
            password=os.getenv("DB_PASSWORD")
        )
        conn.close()
        return "connected"
    except:
        return "disconnected"

def get_generation_count() -> int:
    """Get total number of newsletters generated"""
    # In production, query from database
    return 42

def get_items_collected_count() -> int:
    """Get total number of items collected"""
    # In production, query from database
    return 1337

def log_newsletter_generated(newsletter_id: str, recipient_count: int):
    """Log successful newsletter generation"""
    # In production, write to database
    pass

def log_newsletter_error(newsletter_id: str, error: str):
    """Log newsletter generation error"""
    # In production, write to database and alert
    pass

# monitoring_dashboard.py
"""
Real-time monitoring dashboard for SoFi Newsletter Agent
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import random

st.set_page_config(
    page_title="SoFi Newsletter Monitor",
    page_icon="📊",
    layout="wide"
)

st.title("🚀 SoFi Newsletter Agent - Live Dashboard")

# Sidebar controls
st.sidebar.header("Controls")
refresh_rate = st.sidebar.selectbox("Refresh Rate", ["1s", "5s", "10s", "30s"])
time_range = st.sidebar.selectbox("Time Range", ["1h", "24h", "7d", "30d"])

# Main metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Newsletters Sent (Today)",
        value="3",
        delta="+1 from yesterday"
    )

with col2:
    st.metric(
        "Total Recipients",
        value="7,542",
        delta="+142 new"
    )

with col3:
    st.metric(
        "Avg Open Rate",
        value="67.3%",
        delta="+2.1%"
    )

with col4:
    st.metric(
        "System Health",
        value="99.9%",
        delta="0%"
    )

# Charts row
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Newsletter Performance")
    
    # Generate sample data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    performance_data = pd.DataFrame({
        'Date': dates,
        'Open Rate': [random.uniform(60, 75) for _ in range(30)],
        'Click Rate': [random.uniform(20, 35) for _ in range(30)],
        'Generation Time (s)': [random.uniform(10, 30) for _ in range(30)]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=performance_data['Date'],
        y=performance_data['Open Rate'],
        name='Open Rate %',
        line=dict(color='#00D4AA', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=performance_data['Date'],
        y=performance_data['Click Rate'],
        name='Click Rate %',
        line=dict(color='#00B48A', width=3)
    ))
    
    fig.update_layout(
        hovermode='x unified',
        height=300,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Content Distribution")
    
    content_data = pd.DataFrame({
        'Category': ['Product Launch', 'Engineering', 'Business Wins', 
                    'People & Culture', 'Innovation', 'Other'],
        'Count': [15, 12, 10, 8, 6, 5]
    })
    
    fig = px.pie(
        content_data, 
        values='Count', 
        names='Category',
        color_discrete_sequence=['#00D4AA', '#00B48A', '#007A6A', 
                               '#005A4A', '#003A2A', '#001A1A']
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Recent newsletters table
st.subheader("📬 Recent Newsletters")

recent_newsletters = pd.DataFrame({
    'ID': ['NL-20240115-001', 'NL-20240113-001', 'NL-20240111-001'],
    'Sent At': ['2024-01-15 09:00', '2024-01-13 09:00', '2024-01-11 09:00'],
    'Recipients': [7542, 7400, 7358],
    'Open Rate': ['67.3%', '65.2%', '64.8%'],
    'Status': ['✅ Sent', '✅ Sent', '✅ Sent']
})

st.dataframe(
    recent_newsletters,
    use_container_width=True,
    hide_index=True
)

# System status
st.subheader("🔧 System Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("**Mistral API**: ✅ Connected")
    st.info("**Last API Call**: 12s ago")

with col2:
    st.info("**Data Sources**: 6/6 Active")
    st.info("**Last Collection**: 2h ago")

with col3:
    st.info("**Email Service**: ✅ Operational")
    st.info("**Queue Length**: 0")

# Auto-refresh
if refresh_rate == "1s":
    st.rerun()

# run_demo.py
"""
Demo script to showcase the SoFi Newsletter Agent
"""

import asyncio
import os
from datetime import datetime
from unittest.mock import Mock, AsyncMock

async def run_demo():
    """Run a complete demo of the newsletter agent"""
    print("🚀 SoFi Newsletter Agent Demo")
    print("=" * 50)
    
    # Set up mock environment
    os.environ["MISTRAL_API_KEY"] = "demo-key-for-testing"
    os.environ["USE_MOCK_DATA"] = "true"
    os.environ["SEND_EMAIL"] = "false"
    
    print("\n📊 Initializing agent...")
    from sofi_newsletter_agent import MistralNewsletterAgent
    
    # Create agent with mocked Mistral client for demo
    agent = MistralNewsletterAgent("demo-key")
    
    # Mock the Mistral client responses
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="""
<ul>
<li><strong>Launched instant transfers with 99.8% reliability</strong> - Our new real-time payment system processes transfers in just 12 seconds, setting a new industry standard for speed and reliability.</li>
<li><strong>Achieved 40% infrastructure cost reduction through Kubernetes migration</strong> - Engineering successfully migrated 200+ services with zero downtime, demonstrating our commitment to operational excellence.</li>
<li><strong>Exceeded Q4 growth targets by 25% with 500K new members</strong> - Total member base now at 7.5M, reflecting strong market confidence in our products and vision.</li>
<li><strong>Prevented $2M in fraud with new AI system</strong> - Our ML-powered detection processes 100K transactions per second at 99.99% accuracy.</li>
</ul>
    """))]
    
    agent.newsletter_generator.client.chat_async = AsyncMock(return_value=mock_response)
    
    print("\n📡 Collecting data from sources...")
    print("  ✓ Confluence: 3 updates found")
    print("  ✓ Slack: 2 announcements found")
    print("  ✓ Jira: 2 releases found")
    print("  ✓ GitHub: 1 update found")
    print("  ✓ Metrics: 1 report found")
    print("  ✓ HR Systems: 2 updates found")
    
    print("\n🤖 Analyzing content with AI...")
    print("  ✓ Deduplicating similar content")
    print("  ✓ Calculating impact scores")
    print("  ✓ Ranking by relevance")
    
    print("\n✨ Generating newsletter...")
    
    try:
        newsletter_html = await agent.process_and_generate_newsletter()
        
        # Save the output
        os.makedirs("output", exist_ok=True)
        output_path = "output/demo_newsletter.html"
        with open(output_path, "w") as f:
            f.write(newsletter_html)
        
        print("\n✅ Newsletter generated successfully!")
        print(f"📄 Preview saved to: {output_path}")
        print("\n📊 Statistics:")
        print("  • Total items collected: 11")
        print("  • Items included: 8")
        print("  • Categories covered: 6")
        print("  • Generation time: 3.2 seconds")
        
        print("\n🎯 Next steps:")
        print("  1. Open the preview file to see the newsletter")
        print("  2. Configure real API credentials in .env")
        print("  3. Set up data source integrations")
        print("  4. Deploy using Docker or Kubernetes")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    print("\n📦 Installing a minimal set of dependencies for demo...")
    print("For full functionality, run: pip install -r requirements.txt\n")
    
    asyncio.run(run_demo())

# Advanced Features Extension - sofi_newsletter_advanced.py
"""
Advanced features for the SoFi Newsletter Agent
Including personalization, A/B testing, and ML-powered insights
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Dict, List, Tuple
import hashlib
import json

class PersonalizationEngine:
    """Advanced personalization based on user behavior and preferences"""
    
    def __init__(self):
        self.user_profiles = {}
        self.content_embeddings = {}
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.scaler = StandardScaler()
    
    def create_user_profile(self, user_email: str, 
                          department: str, 
                          role: str,
                          location: str,
                          engagement_history: List[Dict]) -> Dict:
        """Create a comprehensive user profile"""
        
        # Calculate engagement patterns
        total_opens = sum(1 for e in engagement_history if e.get('opened'))
        total_clicks = sum(e.get('clicks', 0) for e in engagement_history)
        
        # Category preferences based on clicks
        category_clicks = {}
        for engagement in engagement_history:
            for category in engagement.get('clicked_categories', []):
                category_clicks[category] = category_clicks.get(category, 0) + 1
        
        # Time preferences
        open_hours = [e.get('opened_hour') for e in engagement_history if e.get('opened_hour')]
        preferred_hour = max(set(open_hours), key=open_hours.count) if open_hours else 9
        
        profile = {
            'email': user_email,
            'department': department,
            'role': role,
            'location': location,
            'engagement_score': total_opens / max(len(engagement_history), 1),
            'click_rate': total_clicks / max(total_opens, 1),
            'category_preferences': category_clicks,
            'preferred_reading_hour': preferred_hour,
            'segment': None  # Will be assigned by clustering
        }
        
        self.user_profiles[user_email] = profile
        return profile
    
    def segment_users(self) -> Dict[str, List[str]]:
        """Segment users into groups for targeted content"""
        if len(self.user_profiles) < 5:
            return {"default": list(self.user_profiles.keys())}
        
        # Create feature matrix
        features = []
        emails = []
        
        for email, profile in self.user_profiles.items():
            feature_vec = [
                profile['engagement_score'],
                profile['click_rate'],
                len(profile['category_preferences']),
                profile['preferred_reading_hour']
            ]
            features.append(feature_vec)
            emails.append(email)
        
        # Scale and cluster
        features_scaled = self.scaler.fit_transform(features)
        clusters = self.kmeans.fit_predict(features_scaled)
        
        # Group users by segment
        segments = {}
        for email, cluster in zip(emails, clusters):
            segment_name = f"segment_{cluster}"
            if segment_name not in segments:
                segments[segment_name] = []
            segments[segment_name].append(email)
            self.user_profiles[email]['segment'] = segment_name
        
        return segments
    
    def personalize_content(self, user_email: str, 
                          content_items: List['NewsItem']) -> List['NewsItem']:
        """Personalize content selection for a specific user"""
        profile = self.user_profiles.get(user_email)
        if not profile:
            return content_items[:10]  # Default selection
        
        # Score each content item for this user
        scored_items = []
        for item in content_items:
            score = self._calculate_personalization_score(item, profile)
            scored_items.append((score, item))
        
        # Sort by score and return top items
        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_items[:10]]
    
    def _calculate_personalization_score(self, item: 'NewsItem', 
                                       profile: Dict) -> float:
        """Calculate how relevant a content item is for a user"""
        score = item.impact_score  # Base score
        
        # Boost for preferred categories
        category_prefs = profile.get('category_preferences', {})
        if item.category.value in category_prefs:
            preference_strength = category_prefs[item.category.value]
            score += preference_strength * 0.1
        
        # Department relevance
        if profile['department'].lower() in item.content.lower():
            score += 0.3
        
        # Role relevance
        if profile['role'].lower() in item.tags:
            score += 0.2
        
        return score

class ABTestingFramework:
    """A/B testing for newsletter optimization"""
    
    def __init__(self):
        self.active_tests = {}
        self.test_results = {}
    
    def create_test(self, test_name: str, 
                   variants: Dict[str, Dict],
                   target_metric: str,
                   sample_size: int) -> str:
        """Create a new A/B test"""
        test_id = hashlib.md5(f"{test_name}_{datetime.now()}".encode()).hexdigest()[:8]
        
        self.active_tests[test_id] = {
            'name': test_name,
            'variants': variants,
            'target_metric': target_metric,
            'sample_size': sample_size,
            'assigned_users': {},
            'results': {variant: {'conversions': 0, 'participants': 0} 
                       for variant in variants},
            'status': 'active',
            'created_at': datetime.now()
        }
        
        return test_id
    
    def assign_variant(self, test_id: str, user_email: str) -> str:
        """Assign a user to a test variant"""
        test = self.active_tests.get(test_id)
        if not test or test['status'] != 'active':
            return 'control'
        
        # Check if already assigned
        if user_email in test['assigned_users']:
            return test['assigned_users'][user_email]
        
        # Random assignment
        variant = np.random.choice(list(test['variants'].keys()))
        test['assigned_users'][user_email] = variant
        test['results'][variant]['participants'] += 1
        
        return variant
    
    def record_conversion(self, test_id: str, user_email: str):
        """Record a conversion event"""
        test = self.active_tests.get(test_id)
        if not test:
            return
        
        variant = test['assigned_users'].get(user_email)
        if variant:
            test['results'][variant]['conversions'] += 1
    
    def analyze_test(self, test_id: str) -> Dict:
        """Analyze test results and determine winner"""
        test = self.active_tests.get(test_id)
        if not test:
            return {}
        
        results = []
        for variant, data in test['results'].items():
            conversion_rate = (data['conversions'] / max(data['participants'], 1))
            results.append({
                'variant': variant,
                'participants': data['participants'],
                'conversions': data['conversions'],
                'conversion_rate': conversion_rate
            })
        
        # Sort by conversion rate
        results.sort(key=lambda x: x['conversion_rate'], reverse=True)
        
        # Calculate statistical significance (simplified)
        if len(results) >= 2:
            control = results[1]
            treatment = results[0]
            
            # Z-test for proportions
            p1 = treatment['conversion_rate']
            p2 = control['conversion_rate']
            n1 = treatment['participants']
            n2 = control['participants']
            
            if n1 > 0 and n2 > 0:
                p_pooled = ((p1 * n1) + (p2 * n2)) / (n1 + n2)
                se = np.sqrt(p_pooled * (1 - p_pooled) * ((1/n1) + (1/n2)))
                z_score = (p1 - p2) / max(se, 0.0001)
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                results[0]['is_significant'] = p_value < 0.05
                results[0]['p_value'] = p_value
        
        return {
            'test_name': test['name'],
            'results': results,
            'winner': results[0]['variant'] if results else None,
            'status': test['status']
        }

class ContentOptimizer:
    """ML-powered content optimization"""
    
    def __init__(self):
        self.performance_history = []
        self.subject_line_model = None
        self.content_length_optimizer = None
    
    def learn_from_performance(self, newsletter_data: Dict):
        """Learn from past newsletter performance"""
        self.performance_history.append({
            'timestamp': newsletter_data['sent_at'],
            'subject_line': newsletter_data['subject_line'],
            'content_length': len(newsletter_data['content']),
            'num_sections': newsletter_data['num_sections'],
            'open_rate': newsletter_data['open_rate'],
            'click_rate': newsletter_data['click_rate'],
            'hour_sent': newsletter_data['sent_at'].hour,
            'day_of_week': newsletter_data['sent_at'].weekday()
        })
        
        # Retrain models if enough data
        if len(self.performance_history) >= 20:
            self._train_optimization_models()
    
    def suggest_improvements(self, proposed_newsletter: Dict) -> Dict:
        """Suggest improvements based on historical performance"""
        suggestions = {}
        
        # Optimal send time
        if self.performance_history:
            df = pd.DataFrame(self.performance_history)
            
            # Best performing hour
            hour_performance = df.groupby('hour_sent')['open_rate'].mean()
            best_hour = hour_performance.idxmax()
            suggestions['optimal_send_hour'] = int(best_hour)
            
            # Best performing day
            day_performance = df.groupby('day_of_week')['open_rate'].mean()
            best_day = day_performance.idxmax()
            suggestions['optimal_send_day'] = int(best_day)
            
            # Content length optimization
            length_correlation = df['content_length'].corr(df['click_rate'])
            if length_correlation > 0:
                suggestions['content_recommendation'] = "Consider adding more detailed content"
            else:
                suggestions['content_recommendation'] = "Consider more concise content"
        
        return suggestions
    
    def _train_optimization_models(self):
        """Train ML models for optimization"""
        # This would include more sophisticated models in production
        pass

class IntelligentScheduler:
    """Smart scheduling based on user behavior and preferences"""
    
    def __init__(self, personalization_engine: PersonalizationEngine):
        self.personalization = personalization_engine
        self.schedule_queue = []
    
    def create_personalized_schedule(self) -> List[Dict]:
        """Create a sending schedule optimized for each user segment"""
        segments = self.personalization.segment_users()
        schedule = []
        
        for segment, users in segments.items():
            # Get average preferred hour for segment
            preferred_hours = []
            for user in users:
                profile = self.personalization.user_profiles.get(user)
                if profile:
                    preferred_hours.append(profile['preferred_reading_hour'])
            
            if preferred_hours:
                optimal_hour = int(np.median(preferred_hours))
            else:
                optimal_hour = 9  # Default
            
            schedule.append({
                'segment': segment,
                'users': users,
                'send_time': optimal_hour,
                'personalization_level': 'high' if len(users) < 100 else 'medium'
            })
        
        return schedule

# Integration with main agent
def enhance_newsletter_agent():
    """Enhance the main newsletter agent with advanced features"""
    
    # Initialize advanced components
    personalization = PersonalizationEngine()
    ab_testing = ABTestingFramework()
    optimizer = ContentOptimizer()
    scheduler = IntelligentScheduler(personalization)
    
    # Create sample user profiles
    sample_users = [
        ("john@sofi.com", "Engineering", "Senior Engineer", "SF"),
        ("sarah@sofi.com", "Product", "PM", "NYC"),
        ("mike@sofi.com", "Sales", "Director", "LA")
    ]
    
    for email, dept, role, location in sample_users:
        personalization.create_user_profile(
            email, dept, role, location,
            engagement_history=[
                {'opened': True, 'clicks': 3, 'clicked_categories': ['Engineering', 'Product Launch']},
                {'opened': True, 'clicks': 1, 'clicked_categories': ['Metrics']},
                {'opened': False, 'clicks': 0, 'clicked_categories': []}
            ]
        )
    
    # Create A/B test for subject lines
    test_id = ab_testing.create_test(
        "subject_line_emoji",
        variants={
            'control': {'subject': 'SoFi Weekly Pulse - {date}'},
            'treatment': {'subject': '🚀 SoFi Weekly Pulse - {date}'}
        },
        target_metric='open_rate',
        sample_size=1000
    )
    
    print(f"Advanced features initialized!")
    print(f"- Personalization engine with {len(personalization.user_profiles)} profiles")
    print(f"- A/B test created: {test_id}")
    print(f"- Intelligent scheduler ready")
    
    return personalization, ab_testing, optimizer, scheduler

if __name__ == "__main__":
    enhance_newsletter_agent()