"""
Curated keyword lists for gazetteer-based extraction (via spaCy's PhraseMatcher)
of skills and education level, since general-purpose NER (spaCy's built-in
entities, or a CoNLL-trained transformer) has no SKILL/EDUCATION category and
will misclassify or ignore these terms - confirmed empirically on real resumes,
e.g. "Machine Learning", "CNN", "Keras" being mistagged as ORG.

Organized by the domains present in the 24-category resume dataset, since a
tech-only list would be useless for e.g. CHEF or AVIATION resumes. Each list
is deliberately flat (not nested) so it can be fed directly into a PhraseMatcher.

NOTE on AMBIGUOUS_SKILLS: short abbreviations like "R", "CV", "AI" are prone to
false positives (e.g. "CV" meaning "curriculum vitae" on a resume, not Computer
Vision). This list only flags which terms are risky - the matcher built on top
of this file needs to handle them with case-sensitive/standalone-token matching
rather than the case-insensitive matching used for everything else.
"""

TECH_DATA_SKILLS = [
    # Programming languages
    "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#",
    "R", "SQL", "PHP", "Ruby", "Go", "Golang", "Kotlin", "Swift",
    "Scala", "MATLAB", "Bash", "Shell Scripting", "PowerShell",

    # AI and machine learning
    "Artificial Intelligence", "AI", "Machine Learning", "ML",
    "Deep Learning", "DL", "Computer Vision", "CV",
    "Natural Language Processing", "NLP",
    "Generative AI", "Large Language Models", "LLM", "LLMs",
    "Prompt Engineering", "Transformers", "BERT", "GPT",
    "Retrieval-Augmented Generation", "RAG",
    "Recommendation Systems", "Predictive Modeling",
    "Supervised Learning", "Unsupervised Learning",
    "Reinforcement Learning", "Transfer Learning",
    "Feature Engineering", "Model Deployment",
    "Model Evaluation", "Hyperparameter Tuning",
    "Time Series Analysis", "Anomaly Detection",

    # Neural network concepts
    "Neural Network", "Artificial Neural Network", "ANN",
    "Convolutional Neural Network", "CNN",
    "Recurrent Neural Network", "RNN",
    "Long Short-Term Memory", "LSTM",
    "Generative Adversarial Network", "GAN",
    "EfficientNet", "ResNet", "YOLO", "Vision Transformer", "ViT",

    # AI and data libraries
    "TensorFlow", "PyTorch", "Keras", "Scikit-learn",
    "OpenCV", "Pandas", "NumPy", "SciPy", "Matplotlib",
    "Seaborn", "Hugging Face", "XGBoost", "LightGBM", "CatBoost", "Ultralytics",

    # Data science and analytics
    "Data Analysis", "Data Analytics", "Data Science",
    "Statistical Analysis", "Statistics", "Data Mining",
    "Data Visualization", "Exploratory Data Analysis", "EDA",
    "Business Intelligence", "BI", "A/B Testing", "Experiment Design",

    # Databases
    "MySQL", "PostgreSQL", "Postgres", "SQL Server",
    "Microsoft SQL Server", "Oracle Database", "SQLite",
    "MongoDB", "Redis", "Cassandra", "DynamoDB",
    "Elasticsearch", "Neo4j", "Firebase", "Supabase",
    "Vector Database", "Pinecone", "FAISS", "ChromaDB", "pgvector",

    # Data engineering
    "ETL", "ELT", "Data Pipeline", "Data Warehousing", "Data Warehouse",
    "Data Lake", "Big Data", "Apache Spark", "Spark", "Hadoop",
    "Apache Kafka", "Kafka", "Airflow", "Apache Airflow",
    "Databricks", "Snowflake", "dbt",

    # Backend and APIs
    "REST API", "RESTful API", "API Development",
    "FastAPI", "Django", "Flask", "Spring Boot",
    "Node.js", "Express.js", "GraphQL", "Microservices",

    # Frontend and mobile
    "React", "Angular", "Vue.js", "HTML", "HTML5", "CSS", "CSS3",
    "Bootstrap", "Flutter", "React Native", "Android Development", "iOS Development",

    # Cloud and DevOps
    "AWS", "Amazon Web Services", "Azure", "Microsoft Azure",
    "Google Cloud", "Google Cloud Platform", "GCP",
    "Docker", "Kubernetes", "Terraform",
    "Jenkins", "GitHub Actions", "CI/CD",
    "DevOps", "MLOps", "MLflow", "Kubeflow", "Linux", "Ubuntu",

    # Version control and development tools
    "Git", "GitHub", "GitLab", "Bitbucket",
    "Jupyter Notebook", "Google Colab", "Visual Studio Code",

    # IT support
    "IT Support", "Technical Support", "Help Desk",
    "Hardware Troubleshooting", "Software Troubleshooting",
    "Network Troubleshooting", "Active Directory",
    "Windows Server", "TCP/IP", "DNS", "DHCP", "VPN",
]

BUSINESS_FINANCE_SKILLS = [
    # Finance and accounting
    "Financial Analysis", "Financial Reporting", "Financial Planning",
    "Financial Modeling", "Budgeting", "Forecasting", "Accounting",
    "General Ledger", "Bookkeeping", "QuickBooks",
    "Accounts Payable", "Accounts Receivable",
    "Payroll", "Auditing", "Internal Audit", "External Audit",
    "Tax Preparation", "Tax Accounting", "Bank Reconciliation",
    "Balance Sheet", "Income Statement", "Cash Flow",
    "Cost Accounting", "Management Accounting", "IFRS", "GAAP",
    "Risk Management", "Credit Analysis", "Investment Analysis",
    "Portfolio Management", "Treasury Management",

    # Sales and business development
    "Sales", "Sales Strategy", "B2B Sales", "B2C Sales",
    "Inside Sales", "Outside Sales", "Retail Sales",
    "Business Development", "Lead Generation", "Prospecting",
    "Cold Calling", "Pipeline Management", "Sales Forecasting",
    "Account Management", "Key Account Management",
    "Client Relationship Management", "CRM", "Salesforce", "HubSpot",
    "Negotiation", "Contract Negotiation", "Closing",
    "Upselling", "Cross-selling",

    # Operations and procurement
    "Operations Management", "Business Operations",
    "Process Improvement", "Process Optimization", "Cost Reduction",
    "Vendor Management", "Supplier Management", "Procurement",
    "Purchasing", "Sourcing", "Strategic Sourcing",
    "Contract Management", "Inventory Management", "Supply Chain Management",

    # Marketing
    "Marketing", "Digital Marketing", "Market Research", "Marketing Strategy",
    "Brand Management", "Product Marketing", "Email Marketing",
    "Content Marketing", "Search Engine Optimization", "SEO",
    "Search Engine Marketing", "SEM", "Google Analytics", "Google Ads",
    "Meta Ads", "Facebook Ads", "Campaign Management",

    # Administration
    "Office Administration", "Administrative Support", "Executive Assistance",
    "Calendar Management", "Meeting Coordination", "Travel Coordination",
    "Record Keeping", "Document Management", "Data Entry", "Report Preparation",
    "Microsoft Office", "Microsoft Excel", "Excel",
    "Microsoft Word", "Microsoft PowerPoint",
]

HR_SKILLS = [
    "Recruitment", "Onboarding", "Employee Relations", "Benefits Administration",
    "Performance Management", "Talent Acquisition", "HRIS", "Compensation",
    "Labor Relations", "Training and Development",
    "Workforce Planning", "Employee Engagement", "Compliance",
]

HEALTHCARE_SKILLS = [
    "Patient Care", "Clinical Research", "Nursing", "CPR", "HIPAA",
    "Medical Billing", "Medical Coding", "Electronic Health Records", "EHR",
    "Phlebotomy", "Vital Signs", "Patient Assessment", "Pharmacology",
    "ICD-9", "ICD-10", "CPT", "Infection Control",
]

CONSTRUCTION_ENGINEERING_SKILLS = [
    "AutoCAD", "Blueprint Reading", "Project Management", "OSHA",
    "Structural Engineering", "Civil Engineering", "Solidworks",
    "Quality Control", "Scheduling", "Cost Estimation", "Site Safety",
    "Welding", "Electrical Systems", "HVAC", "Plumbing", "Carpentry",
]

CULINARY_SKILLS = [
    "Menu Planning", "Food Safety", "ServSafe", "Culinary Arts",
    "Inventory Management", "Food Preparation", "Kitchen Management",
    "Catering", "Food Cost Control", "Sanitation", "Baking", "Pastry",
]

AVIATION_LOGISTICS_SKILLS = [
    "FAA Regulations", "Aircraft Maintenance", "Logistics", "Flight Operations",
    "Supply Chain", "Inventory Control", "Shipping", "Warehouse Management",
    "AOG", "Purchasing", "Quality Assurance", "Expediting",
]

DESIGN_ARTS_MEDIA_SKILLS = [
    "Adobe Photoshop", "Adobe Illustrator", "Adobe InDesign", "Graphic Design",
    "UI/UX", "Figma", "Content Creation", "Social Media Marketing",
    "Video Editing", "Copywriting", "Branding", "Typography", "Web Design",
]

BPO_CUSTOMER_SUPPORT_SKILLS = [
    # Core customer support
    "Customer Service", "Customer Support", "Customer Care", "Client Support",
    "Technical Support", "Help Desk", "IT Help Desk", "Service Desk",
    "Call Center", "Contact Center", "BPO", "Business Process Outsourcing",

    # Communication channels
    "Inbound Calls", "Outbound Calls", "Call Handling", "Phone Support",
    "Voice Support", "Non-Voice Support", "Email Support", "Chat Support",
    "Live Chat Support", "Omnichannel Support", "Social Media Support",

    # Customer interaction
    "Complaint Resolution", "Customer Retention", "Customer Escalation",
    "Escalation Management", "De-escalation", "Active Listening", "Empathy",
    "Rapport Building", "Upselling", "Cross-selling", "Lead Generation",
    "Appointment Setting", "Telemarketing",

    # Call-center metrics
    "AHT", "Average Handle Time", "Average Handling Time",
    "FCR", "First Call Resolution",
    "CSAT", "Customer Satisfaction", "Customer Satisfaction Score",
    "NPS", "Net Promoter Score", "SLA", "Service Level Agreement",
    "Quality Assurance", "Quality Monitoring", "Call Monitoring",
    "Call Auditing", "KPI", "Key Performance Indicators",
    "Workforce Management", "WFM",

    # Tools
    "Zendesk", "Freshdesk", "ServiceNow", "Salesforce Service Cloud",
    "HubSpot", "Intercom", "Five9", "Genesys", "Avaya", "Cisco Finesse",
    "Zoho CRM", "Microsoft Dynamics 365", "Ticketing System", "CRM Software",

    # Operational skills
    "Ticket Management", "Case Management", "Order Processing",
    "Refund Processing", "Account Verification", "Data Entry",
    "Documentation", "Knowledge Base", "Troubleshooting",
    "Remote Support", "Back Office Support",
]

EDUCATION_TEACHING_SKILLS = [
    "Teaching", "Classroom Management", "Lesson Planning",
    "Curriculum Development", "Curriculum Design",
    "Student Assessment", "Educational Technology",
    "Instructional Design", "Tutoring", "Mentoring",
    "Special Education", "Early Childhood Education",
    "E-learning", "Learning Management System", "LMS",
    "Blackboard", "Moodle", "Google Classroom",
]

LEGAL_SKILLS = [
    "Legal Research", "Legal Writing", "Litigation",
    "Contract Drafting", "Contract Review", "Case Management",
    "Legal Compliance", "Corporate Law", "Commercial Law",
    "Civil Law", "Criminal Law", "Intellectual Property",
    "Due Diligence", "Document Review", "Paralegal", "Westlaw", "LexisNexis",
]

SECURITY_SKILLS = [
    "Security Operations", "Physical Security", "Access Control",
    "Surveillance", "CCTV", "Incident Reporting", "Risk Assessment",
    "Emergency Response", "Loss Prevention", "Cybersecurity",
    "Information Security", "Network Security", "Penetration Testing",
    "Vulnerability Assessment", "SIEM", "SOC", "Security Operations Center",
    "Firewalls", "Incident Response",
]

AUTOMOTIVE_MECHANICAL_SKILLS = [
    "Automotive Repair", "Vehicle Maintenance", "Preventive Maintenance",
    "Mechanical Engineering", "Mechanical Maintenance", "Diagnostics",
    "Engine Repair", "Brake Repair", "Transmission Repair",
    "Hydraulics", "Pneumatics", "CNC", "Machining",
    "Technical Drawing", "SolidWorks", "CATIA",
]

FITNESS_SPORTS_SKILLS = [
    "Personal Training", "Fitness Training", "Strength Training",
    "Cardiovascular Training", "Exercise Programming", "Sports Coaching",
    "Nutrition Coaching", "Group Fitness", "First Aid",
    "Injury Prevention", "Body Composition Assessment",
]

RETAIL_HOSPITALITY_SKILLS = [
    "Retail Operations", "Point of Sale", "POS", "Cash Handling",
    "Merchandising", "Visual Merchandising", "Store Management",
    "Hospitality", "Hotel Operations", "Front Desk", "Guest Relations",
    "Reservations", "Housekeeping", "Food and Beverage", "Event Planning",
]

MANUFACTURING_SKILLS = [
    "Manufacturing", "Production Planning", "Production Management",
    "Lean Manufacturing", "Six Sigma", "Kaizen", "5S",
    "Quality Management", "Quality Control", "Root Cause Analysis",
    "CAPA", "GMP", "ISO 9001", "Assembly Line",
]

SOFT_SKILLS = [
    "Leadership", "Communication", "Teamwork", "Problem Solving",
    "Time Management", "Critical Thinking", "Adaptability",
    "Multitasking", "Attention to Detail",
    "Public Speaking", "Conflict Resolution", "Decision Making",
]

EDUCATION_LEVELS = [
    # Doctoral
    "PhD", "Ph.D.", "Doctorate", "Doctoral Degree", "Doctor of Philosophy",
    "DBA", "Doctor of Business Administration",
    "EdD", "Ed.D.", "Doctor of Education",
    "MD", "M.D.", "Doctor of Medicine", "Juris Doctor", "JD", "J.D.",

    # Master's
    "Master's Degree", "Master of Science", "MSc", "M.S.",
    "Master of Arts", "M.A.",
    "Master of Engineering", "MEng", "M.Eng.",
    "Master of Technology", "MTech", "M.Tech.",
    "MBA", "Master of Business Administration",
    "Postgraduate Degree", "Postgraduate Diploma", "PGDip",

    # Bachelor's
    "Bachelor's Degree", "Bachelor of Science", "BSc", "B.S.",
    "Bachelor of Arts", "B.A.",
    "Bachelor of Engineering", "BEng", "B.Eng.",
    "Bachelor of Technology", "BTech", "B.Tech.",
    "Bachelor of Business Administration", "BBA", "B.B.A.",
    "Bachelor of Commerce", "BCom", "B.Com.", "Undergraduate Degree",

    # Associate
    "Associate Degree", "Associate's Degree",
    "Associate of Science", "A.S.", "Associate of Arts", "A.A.",
    "Associate of Applied Science", "AAS", "A.A.S.",

    # Secondary education
    "High School Diploma", "Secondary School Certificate",
    "Secondary Education", "GED", "General Educational Development",
    "A Levels", "A-Levels", "GCSE", "IGCSE",

    # Vocational
    "Diploma", "Technical Diploma", "Vocational Diploma",
    "Professional Diploma", "Certificate", "Certification",
    "Professional Certificate", "Technical Certificate",
    "Trade School", "Vocational Training",
]

# Short abbreviations that risk false positives with plain case-insensitive
# matching (e.g. lowercase "cv" meaning "curriculum vitae", not Computer
# Vision). The matcher built on top of this file should handle these with
# case-sensitive/standalone-token matching, not the default case-insensitive
# pass used for everything else.
AMBIGUOUS_SKILLS = {"R", "C", "AI", "CV", "ML", "DL", "BI"}

# Maps common variants/abbreviations to one canonical form, so e.g. "GCP",
# "Google Cloud", and "Google Cloud Platform" aren't treated as three
# unrelated skills in downstream output.
SKILL_ALIASES = {
    "gcp": "Google Cloud Platform",
    "google cloud": "Google Cloud Platform",
    "google cloud platform": "Google Cloud Platform",
    "aws": "Amazon Web Services",
    "amazon web services": "Amazon Web Services",
    "ai": "Artificial Intelligence",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "dl": "Deep Learning",
    "deep learning": "Deep Learning",
    "cv": "Computer Vision",
    "computer vision": "Computer Vision",
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "bi": "Business Intelligence",
    "business intelligence": "Business Intelligence",
    "cnn": "Convolutional Neural Network",
    "convolutional neural network": "Convolutional Neural Network",
    "rnn": "Recurrent Neural Network",
    "recurrent neural network": "Recurrent Neural Network",
    "lstm": "Long Short-Term Memory",
    "long short-term memory": "Long Short-Term Memory",
    "gan": "Generative Adversarial Network",
    "generative adversarial network": "Generative Adversarial Network",
    "llm": "Large Language Models",
    "llms": "Large Language Models",
    "large language models": "Large Language Models",
    "rag": "Retrieval-Augmented Generation",
    "retrieval-augmented generation": "Retrieval-Augmented Generation",
    "vit": "Vision Transformer",
    "vision transformer": "Vision Transformer",
    "eda": "Exploratory Data Analysis",
    "exploratory data analysis": "Exploratory Data Analysis",
    "sklearn": "Scikit-learn",
    "scikit learn": "Scikit-learn",
    "scikit-learn": "Scikit-learn",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "ms excel": "Microsoft Excel",
    "microsoft excel": "Microsoft Excel",
    "excel": "Microsoft Excel",
    "customer care": "Customer Service",
    "customer support": "Customer Service",
    "customer service": "Customer Service",
    "average handle time": "Average Handling Time",
    "average handling time": "Average Handling Time",
    "aht": "Average Handling Time",
    "first call resolution": "First Call Resolution",
    "fcr": "First Call Resolution",
    "customer satisfaction score": "Customer Satisfaction Score",
    "customer satisfaction": "Customer Satisfaction Score",
    "csat": "Customer Satisfaction Score",
}


def canonicalize_skill(skill: str) -> str:
    """
    Map a matched skill to its canonical form.

    Checks SKILL_ALIASES first (for abbreviations/variants that should
    collapse to a different canonical name), then falls back to the
    properly-cased version already defined in ALL_SKILLS (so a skill matched
    in whatever casing the resume happened to use, e.g. "feature engineering",
    still comes back as "Feature Engineering" instead of passing through
    unchanged). _CANONICAL_CASING is built after ALL_SKILLS at the bottom of
    this module; referencing it here is fine since it only needs to exist by
    call time, not by the time this function is defined.
    """
    key = skill.strip().lower()
    if key in SKILL_ALIASES:
        return SKILL_ALIASES[key]
    if key in _CANONICAL_CASING:
        return _CANONICAL_CASING[key]
    return skill.strip()


def deduplicate_phrases(phrases: list[str]) -> list[str]:
    """Remove exact duplicates (case-insensitively) while preserving order."""
    seen = set()
    result = []
    for phrase in phrases:
        cleaned = phrase.strip()
        normalized = cleaned.casefold()
        if cleaned and normalized not in seen:
            seen.add(normalized)
            result.append(cleaned)
    return result


SKILL_GROUPS = {
    "technical": TECH_DATA_SKILLS,
    "business_finance": BUSINESS_FINANCE_SKILLS,
    "human_resources": HR_SKILLS,
    "healthcare": HEALTHCARE_SKILLS,
    "construction_engineering": CONSTRUCTION_ENGINEERING_SKILLS,
    "culinary": CULINARY_SKILLS,
    "aviation_logistics": AVIATION_LOGISTICS_SKILLS,
    "design_media": DESIGN_ARTS_MEDIA_SKILLS,
    "bpo_customer_support": BPO_CUSTOMER_SUPPORT_SKILLS,
    "education_teaching": EDUCATION_TEACHING_SKILLS,
    "legal": LEGAL_SKILLS,
    "security": SECURITY_SKILLS,
    "automotive_mechanical": AUTOMOTIVE_MECHANICAL_SKILLS,
    "fitness_sports": FITNESS_SPORTS_SKILLS,
    "retail_hospitality": RETAIL_HOSPITALITY_SKILLS,
    "manufacturing": MANUFACTURING_SKILLS,
    "soft_skills": SOFT_SKILLS,
}

ALL_SKILLS = deduplicate_phrases(
    [skill for group in SKILL_GROUPS.values() for skill in group]
)

# Case-insensitive lookup used by canonicalize_skill() to restore proper
# casing for skills that don't have an explicit entry in SKILL_ALIASES.
_CANONICAL_CASING = {s.lower(): s for s in ALL_SKILLS}
