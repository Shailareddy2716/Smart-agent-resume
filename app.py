import os, json
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic

app = Flask(__name__)
CORS(app, origins="*", allow_headers=["Content-Type"], methods=["GET","POST","OPTIONS"])

@app.after_request
def add_cors(r):
    r.headers["Access-Control-Allow-Origin"] = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return r

@app.route("/tailor", methods=["OPTIONS"])
def preflight(): return "", 204

PROFILE = """
CANDIDATE: Shaila Reddy Kankanala
Contact: 925-819-3509 | shailareddy1603@gmail.com | linkedin.com/in/shailareddy | San Jose, CA

EDUCATION:
1. Master of Science in Applied Data Intelligence | San Jose State University | San Jose, CA | Aug. 2024 - May 2026
2. Bachelor of Technology in Information Technology | Jawaharlal Nehru Technological University | Hyderabad, India | Jul. 2018 - Jul. 2022

EXPERIENCE:
[AllyIn.AI] AI Engineer Intern | May 2025 - Aug. 2025 | San Jose, CA
B1: Built data pipelines extracting model training outputs into FAISS and Pinecone vector storage, enabling sub-10s query latency and 35% reduction in LLM hallucinations across 3 enterprise deployments.
B2: Developed FastAPI-based APIs and Python libraries for ML researchers to benchmark LLaMA fine-tuning outputs across 5 task categories, raising model accuracy from 72% to 90%.
B3: Shipped LLM evaluation and data quality tooling used daily by research teams; identified 3 critical dataset issues and lifted benchmark quality rating by 22% within 4 weeks.
B4: Consolidated 4 enterprise data sources into unified Pinecone vector store via Docker, boosting retrieval quality 30% and cutting deployment setup time by 60%.
B5: Prototyped sensor-to-inference event pipeline for emotion-aware AI agent, cutting architecture iteration cycles by 40% and accelerating time-to-demo by 2 weeks.

[Dextara Datamatics] Data Engineer | Aug. 2022 - Jul. 2024 | Hyderabad, India
B1: Engineered ETL/ELT pipelines in Python and SQL on Snowflake and MySQL processing 100K+ monthly transactions; consolidated CRM/ERP reporting across 5+ teams, cutting manual effort by 40%.
B2: Deployed Apache Kafka real-time streaming pipelines shrinking data freshness SLA from 4 hrs to under 30 min for downstream analytics teams.
B3: Operated Apache Airflow pipelines via GCP Cloud Composer managing scheduling, retry logic, alerting, and lineage tracking; saved 8 hrs/week across 20+ active DAGs.
B4: Authored dbt transformation models on Snowflake; built schema drift detection across 3 source systems, improving debugging time by 35% and preventing production incidents.
B5: Migrated 500K+ records integrating Salesforce, Acumatica, and Slack via REST APIs on GCP achieving 98% data consistency; mentored junior engineers on pipeline best practices.
B6: Embedded with US enterprise client as primary technical liaison; built Tableau KPI dashboards across 5+ teams and unblocked 3 delayed deliverables under delivery pressure.
B7: Designed data quality check frameworks across ingestion and transformation layers, reducing downstream reporting errors by 30% and improving SLA adherence across 4 business units.

[RineX.AI] Data Analyst Intern | Sept. 2021 - Jul. 2022 | Hyderabad, India
B1: Queried 1M+ row datasets via SQL and Python (Pandas, NumPy) to surface business KPIs; automated reporting workflows saving 12 hrs/week and cutting turnaround from 3 days to same-day.
B2: Identified data quality gaps across 6 source tables; built Python cleansing scripts cutting downstream report errors by 45% and saving 4 hrs of QA per cycle.
B3: Built Tableau and Power BI dashboards adopted across 4 teams; co-led churn analysis contributing to 15% retention uplift and monthly churn drop from 8% to 4.5% over 2 quarters.
B4: Developed data dictionaries and onboarding playbooks; onboarded 3 analysts and cut ramp-up time from 3 weeks to under 10 days.
B5: Collaborated with product and engineering teams to define KPI frameworks; translated findings into executive dashboards adopted by 3 business units.

PROJECTS:
[Retail Demand Forecasting: ELT Pipeline] PySpark, Airflow, Snowflake, dbt, GCP, Docker, Apache Superset, Parquet | 2025
B1: Built end-to-end ELT pipeline on Snowflake ingesting 3 external APIs with dbt mart models and 5 Airflow DAGs; cut forecast generation time by 50% and pipeline failures by 80%.
B2: Modeled time-series demand indices across 20+ product categories; built Apache Superset dashboards across 10+ states, cutting report prep from 2 days to 30 min.

[Plate Planner: RAG Pipeline with Vector Storage] FastAPI, FAISS, Pinecone, Neo4j, SentenceTransformers, Docker | 2025
B1: Built data pipeline ingesting 223K+ recipes into FAISS vector storage enabling generative AI recommendations under 100ms latency; exposed inference via clean REST API endpoints.
B2: Designed graph-based ingredient substitution engine in Neo4j with Word2Vec embeddings; achieved 70% Hit@5 accuracy and 20% quality gain, deployed end-to-end with Docker Compose.

[Inklude: Conditional Generative Model for Air-Drawing] PyTorch, CNN-BiLSTM, Pix2Pix GAN, Stable Diffusion, MediaPipe | 2024
B1: Developed hybrid CNN-BiLSTM-Attention classifier in PyTorch achieving 89.75% accuracy across 45 categories; introduced Motion Language Matrix representation for robust real-time sketch recognition.
B2: Trained 16 class-specific Pix2Pix generators on 16,000 curated image pairs; delivered complete real-time inference pipeline under 100ms end-to-end latency.

SKILLS:
Languages & Databases: Python, SQL, MySQL, Neo4j, REST APIs
Data Engineering: ETL/ELT Pipelines, PySpark, Apache Kafka, Airflow, dbt, Snowflake, Parquet, Data Modeling, Real-Time Streaming
Cloud & Infrastructure: GCP (Cloud Composer, Cloud Storage), Docker, FastAPI, Git, Linux
Visualization: Tableau, Power BI, Apache Superset, Streamlit
AI & ML: PyTorch, Scikit-learn, LLM Fine-Tuning, RAG Pipelines, FAISS, Pinecone, SentenceTransformers, Vector Databases
"""

LATEX_HEADER = r"""\documentclass[letterpaper,10pt]{article}
\usepackage{latexsym}
\usepackage[empty]{fullpage}
\usepackage{titlesec}
\usepackage{marvosym}
\usepackage[usenames,dvipsnames]{color}
\usepackage{verbatim}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\usepackage[english]{babel}
\usepackage{tabularx}
\input{glyphtounicode}
\pagestyle{fancy}\fancyhf{}\fancyfoot{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
\addtolength{\oddsidemargin}{-0.6in}
\addtolength{\evensidemargin}{-0.6in}
\addtolength{\textwidth}{1.2in}
\addtolength{\topmargin}{-.65in}
\addtolength{\textheight}{1.3in}
\urlstyle{same}\raggedbottom\raggedright\setlength{\tabcolsep}{0in}
\titleformat{\section}{\vspace{-8pt}\scshape\raggedright\large}{}{0em}{}[\color{black}\titlerule\vspace{-7pt}]
\pdfgentounicode=1
\newcommand{\resumeItem}[1]{\item\small{#1}}
\newcommand{\resumeSubheading}[4]{
  \vspace{-4pt}\item
    \begin{tabular*}{0.97\textwidth}[t]{l@{\extracolsep{\fill}}r}
      \textbf{#1} & #2 \\
      \textit{\small#3} & \textit{\small #4} \\
    \end{tabular*}\vspace{-9pt}
}
\newcommand{\resumeProjectHeading}[2]{
  \item
    \begin{tabular*}{0.97\textwidth}{l@{\extracolsep{\fill}}r}
      \small#1 & #2 \\
    \end{tabular*}\vspace{-9pt}
}
\renewcommand\labelitemii{$\vcenter{\hbox{\tiny$\bullet$}}$}
\newcommand{\resumeSubHeadingListStart}{\begin{itemize}[leftmargin=0.15in,label={}]}
\newcommand{\resumeSubHeadingListEnd}{\end{itemize}}
\newcommand{\resumeItemListStart}{\begin{itemize}[leftmargin=0.2in,itemsep=0pt,topsep=2pt,parsep=0pt]}
\newcommand{\resumeItemListEnd}{\end{itemize}\vspace{-6pt}}
\begin{document}
"""

def esc(s):
    """Escape special LaTeX characters."""
    for ch, rep in [('\\','__BS__'),('%',r'\%'),('$',r'\$'),('&',r'\&'),('#',r'\#'),('_',r'\_'),('^',r'\textasciicircum{}'),('~',r'\textasciitilde{}')]:
        s = s.replace(ch, rep)
    s = s.replace('__BS__', r'\textbackslash{}')
    return s

def build_latex(data):
    tex = LATEX_HEADER
    # Heading
    tex += r"""\begin{center}
    \textbf{\Huge \scshape Shaila Reddy Kankanala} \\ \vspace{1pt}
    \small 925-819-3509 $|$
    \href{mailto:shailareddy1603@gmail.com}{\underline{shailareddy1603@gmail.com}} $|$
    \href{https://linkedin.com/in/shailareddy}{\underline{linkedin.com/in/shailareddy}} $|$
    San Jose, CA
\end{center}
"""
    # Education
    tex += r"""\section{Education}
  \resumeSubHeadingListStart
    \resumeSubheading{San Jose State University}{Aug. 2024 -- May 2026}{Master of Science in Applied Data Intelligence}{San Jose, CA}
    \resumeSubheading{Jawaharlal Nehru Technological University}{Jul. 2018 -- Jul. 2022}{Bachelor of Technology in Information Technology}{Hyderabad, India}
  \resumeSubHeadingListEnd
"""
    # Experience
    tex += "\\section{Experience}\n  \\resumeSubHeadingListStart\n"
    for exp in data.get("experience", []):
        tex += f"    \\resumeSubheading{{{esc(exp['company'])}}}{{{esc(exp['dates'])}}}{{{esc(exp['title'])}}}{{{esc(exp['location'])}}}\n"
        tex += "      \\resumeItemListStart\n"
        for b in exp.get("bullets", []):
            tex += f"        \\resumeItem{{{esc(b)}}}\n"
        tex += "      \\resumeItemListEnd\n"
    tex += "  \\resumeSubHeadingListEnd\n"

    # Projects
    tex += "\\section{Projects}\n    \\resumeSubHeadingListStart\n"
    for proj in data.get("projects", []):
        tex += f"      \\resumeProjectHeading{{\\textbf{{{esc(proj['title'])}}} $|$ \\emph{{{esc(proj['stack'])}}}}}{{{proj['date']}}}\n"
        tex += "          \\resumeItemListStart\n"
        for b in proj.get("bullets", []):
            tex += f"            \\resumeItem{{{esc(b)}}}\n"
        tex += "          \\resumeItemListEnd\n"
    tex += "    \\resumeSubHeadingListEnd\n"

    # Skills
    tex += "\\section{Technical Skills}\n \\begin{itemize}[leftmargin=0.15in,label={},itemsep=0pt,topsep=0pt]\n    \\small{\\item{\n"
    lines = []
    for sk in data.get("skills", []):
        lines.append(f"     \\textbf{{{esc(sk['category'])}}}{{: {esc(sk['items'])}}}")
    tex += " \\\\\n".join(lines) + "\n    }}\n \\end{itemize}\n\\end{document}\n"
    return tex

def make_prompt(jd):
    return f"""You are an expert ATS resume writer. Tailor Shaila's resume for this job description.

RULES:
- AllyIn.AI: exactly 5 bullets
- Dextara Datamatics: exactly 7 bullets  
- RineX.AI: exactly 5 bullets
- Exactly 2 most relevant projects, 2 bullets each
- All dates/companies/titles exactly as in profile
- Each bullet MAX 18 words. Short and punchy. Must fit on 1 line when printed.
- STAR format: action verb + what + metric. No semicolons joining two full sentences.
- Mirror JD keywords naturally. Never invent facts. No pronouns.
- Skills: only JD-relevant, each category short (fits 1 line)

PROFILE:\n{PROFILE}\n\nJOB DESCRIPTION:\n{jd}

Return ONLY raw JSON:
{{"score":85,"role":"Title","fit":"sentence","matched_keywords":["k1"],"missing_keywords":["k2"],"tips":"1. tip\\n2. tip\\n3. tip","experience":[{{"company":"AllyIn.AI","title":"AI Engineer Intern","dates":"May 2025 -- Aug. 2025","location":"San Jose, CA","bullets":["b1","b2","b3","b4","b5"]}},{{"company":"Dextara Datamatics","title":"Data Engineer","dates":"Aug. 2022 -- Jul. 2024","location":"Hyderabad, India","bullets":["b1","b2","b3","b4","b5","b6","b7"]}},{{"company":"RineX.AI","title":"Data Analyst Intern","dates":"Sept. 2021 -- Jul. 2022","location":"Hyderabad, India","bullets":["b1","b2","b3","b4","b5"]}}],"projects":[{{"title":"exact title from profile","stack":"exact stack from profile","date":"2025","bullets":["b1","b2"]}},{{"title":"exact title from profile","stack":"exact stack from profile","date":"2025","bullets":["b1","b2"]}}],"skills":[{{"category":"Languages & Databases","items":"Python, SQL, MySQL, Neo4j, REST APIs"}},{{"category":"AI & ML Frameworks","items":"..."}},{{"category":"Data Engineering","items":"..."}},{{"category":"Cloud & Infrastructure","items":"..."}},{{"category":"Visualization","items":"..."}}]}}"""

@app.route("/health", methods=["GET"])
def health(): return jsonify({"status": "ok"})

@app.route("/tailor", methods=["POST"])
def tailor():
    try:
        body    = request.get_json()
        jd      = body.get("jd","").strip()
        api_key = body.get("api_key","").strip()
        if not jd:      return jsonify({"error":"No job description"}),400
        if not api_key: return jsonify({"error":"No API key"}),400

        client  = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(model="claude-sonnet-4-5", max_tokens=4000,
                    messages=[{"role":"user","content":make_prompt(jd)}])
        raw   = message.content[0].text
        clean = raw.replace("```json","").replace("```","").strip()
        s,e   = clean.find("{"), clean.rfind("}")
        if s!=-1 and e!=-1: clean=clean[s:e+1]
        data = json.loads(clean)

        # Return tailored data + LaTeX source (frontend compiles via free API)
        latex_src = build_latex(data)
        return jsonify({
            "score":            data.get("score",0),
            "role":             data.get("role",""),
            "fit":              data.get("fit",""),
            "matched_keywords": data.get("matched_keywords",[]),
            "missing_keywords": data.get("missing_keywords",[]),
            "tips":             data.get("tips",""),
            "latex":            latex_src
        })
    except json.JSONDecodeError as e:
        return jsonify({"error":f"Could not parse AI response: {e}. Try again."}),500
    except Exception as e:
        return jsonify({"error":str(e)}),500

if __name__=="__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
