# RetailMind Product Intelligence Agent

**Mid-Term Exam — Building AI Agents (Set-B)**  
**Course:** UG-DSAI | Deploying AI Agents & Workflow Automation

---

## Overview

An AI-powered Product Intelligence Agent for **StyleCraft** (a D2C fashion brand), built for RetailMind Analytics. The agent answers natural language questions about the product catalog, surfaces critical alerts, and generates a daily briefing — all via a conversational Streamlit interface.

---

## Agentic Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│         LLM Router (GPT-4o)             │
│  Classifies intent via function-calling │
│  — NOT keyword/regex matching           │
└────────────┬────────────────────────────┘
             │
    ┌────────┴─────────┐
    │  Tool Dispatcher │
    └────────┬─────────┘
             │
   ┌─────────┴──────────┐
   │   6 Tool Functions  │
   ├────────────────────┤
   │ search_products     │← CATALOG queries
   │ get_inventory_health│← INVENTORY queries
   │ get_pricing_analysis│← PRICING/MARGIN queries
   │ get_review_insights │← REVIEW/SENTIMENT queries
   │ get_category_perf.  │← CATEGORY overview queries
   │ generate_restock_   │← RESTOCK ALERT queries
   │   alert             │
   └─────────┬──────────┘
             │
    ┌────────▼─────────┐
    │  LLM synthesises  │
    │  tool results +   │
    │  conversation     │
    │  memory → reply   │
    └───────────────────┘
```

**Key design choices:**
- **Router Pattern**: GPT-4o uses OpenAI function-calling (tool_choice="auto") to classify intent and select tools. Zero keyword matching.
- **Agentic Loop**: Agent keeps calling tools until LLM produces a final text response.
- **Conversation Memory**: Full message history maintained in session state across turns.
- **Review Caching**: LLM-generated review insights are cached in-memory to avoid redundant API calls.
- **LLM Parameters**: `temperature=0.4` (factual but readable), `top_p=0.95`, `max_tokens=1500` for main agent; `temperature=0.3` for review summarisation (higher factual accuracy).

---

## Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/retailmind-agent-<roll_number>
cd retailmind-agent-<roll_number>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

**.env file format:**
```
OPENAI_API_KEY=sk-...your-key-here...
```

### 4. Place dataset files
Ensure these files are in the project root:
```
retailmind_products.csv
retailmind_reviews.csv
```

### 5. Run the app
```bash
python run.py
```
Or directly:
```bash
streamlit run app.py
```

The Streamlit app will open at `http://localhost:8501`

---

## Features

| Feature | Description |
|---|---|
| **Daily Briefing** | Auto-generated on startup: top 3 critical stock items, worst-rated product, lowest margin flag |
| **6 Specialised Tools** | Inventory health, pricing analysis, review insights, product search, category performance, restock alerts |
| **LLM Router** | GPT-4o classifies intent and selects tools — no hardcoded routing |
| **Multi-turn Memory** | Full conversation context maintained across turns |
| **Category Filter** | Sidebar filter scopes queries to a specific category |
| **Catalog Summary** | Always-visible metrics: total SKUs, critical stock count, avg margin, avg rating |
| **Suggested Prompts** | Clickable example queries on the empty chat screen |

---

## Tool Inventory

| Tool | Route | Description |
|---|---|---|
| `search_products(query, category)` | CATALOG | Text search + category filter, top 5 results |
| `get_inventory_health(product_id)` | INVENTORY | Days to stockout, Critical/Low/Healthy status |
| `get_pricing_analysis(product_id)` | PRICING | Gross margin %, price positioning, low-margin flag |
| `get_review_insights(product_id)` | REVIEWS | LLM-generated sentiment summary + themes |
| `get_category_performance(category)` | CATALOG | Aggregated SKU, margin, stock, revenue metrics |
| `generate_restock_alert(threshold_days)` | INVENTORY | All at-risk products sorted by urgency |

---

## Prohibited
- No cloud deployment
- No external databases  
- API keys never committed (use `.env`)
