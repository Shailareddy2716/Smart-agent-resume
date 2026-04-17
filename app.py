import os, json, subprocess, tempfile, shutil, re
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import anthropic

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": ["Content-Type"], "methods": ["GET","POST","OPTIONS"]}})

# ── Full candidate profile ─────────────────────────────────────────────
PROFILE = """
CANDIDATE: Shaila Reddy Kankanala
Contact: 925-819-3509 | shailareddy1603@gmail.com | linkedin.com/in/shailareddy | San Jose, CA

EDUCATION:
1. Master of Science in Applied Data Intelligence | San Jose State University | San Jose, CA | Aug. 2024 – May 2026
2. Bachelor of Technology in Information Technology | Jawaharlal Nehru Technological University | Hyderabad, India | Jul. 2018 – Jul. 2022

EXPERIENCE:
[AllyIn.AI] AI Engineer Intern | May 2025 – Aug. 2025 | San Jose, CA
B1: Built data pipelines extracting model training outputs into FAISS and Pinecone vector storage systems, enabling sub-10s query latency and 35% reduction in LLM hallucinations across 3 enterprise deployments.
B2: Developed FastAPI-based APIs and Python libraries for ML researchers to manage, explore, and benchmark LLaMA fine-tuning outputs across 5 task categories, raising model accuracy from 72% to 90%.
B3: Shipped LLM evaluation and data quality tooling used daily by research teams; identified 3 critical dataset issues and lifted benchmark quality rating by 22% within 4 weeks.
B4: Consolidated 4 enterprise data sources into unified Pinecone vector store via Docker, boosting retrieval quality 30% and cutting deployment setup time by 60% through reusable infrastructure.
B5: Prototyped sensor-to-inference event pipeline for emotion-aware AI agent, cutting architecture iteration cycles by 40% and accelerating time-to-demo by 2 weeks.

[Dextara Datamatics] Data Engineer | Aug. 2022 – Jul. 2024 | Hyderabad, India
B1: Engineered large-scale ETL/ELT pipelines in Python and SQL on Snowflake and MySQL processing 100K+ monthly transactions; consolidated CRM/ERP reporting across 5+ teams, cutting manual effort by 40%.
B2: Deployed Apache Kafka real-time streaming pipelines for high-volume event ingestion with fault tolerance, shrinking data freshness SLA from 4 hrs to under 30 min and enabling fast queries on time-series data.
B3: Operated Apache Airflow pipelines via GCP Cloud Composer managing scheduling, retry logic, alerting, and lineage tracking in Docker; saved 8 hrs/week and improved observability across 20+ active DAGs.
B4: Authored dbt transformation models on Snowflake; built data cataloging and schema drift detection across 3 source systems, improving debugging time by 35% and preventing production incidents.
B5: Migrated 500K+ records integrating Salesforce, Acumatica, and Slack via REST APIs on GCP achieving 98% data consistency; mentored junior engineers on pipeline best practices.
B6: Embedded with US enterprise client as primary technical liaison; gathered requirements, built Tableau KPI dashboards across 5+ teams, and unblocked 3 delayed deliverables under delivery pressure.
B7: Designed data quality and provenance check frameworks across ingestion and transformation layers, reducing downstream reporting errors by 30% and improving SLA adherence across 4 business units.

[RineX.AI] Data Analyst Intern | Sept. 2021 – Jul. 2022 | Hyderabad, India
B1: Queried 1M+ row datasets via SQL and Python (Pandas, NumPy) to surface business KPIs; automated reporting workflows saving 12 hrs/week and cutting turnaround from 3 days to same-day.
B2: Identified data quality gaps across 6 source tables; built Python cleansing scripts cutting downstream report errors by 45% and saving 4 hrs of QA per cycle.
B3: Built Tableau and Power BI dashboards adopted across 4 teams; co-led churn analysis contributing to 15% retention uplift and monthly churn drop from 8% to 4.5% over 2 quarters.
B4: Developed dataset management tooling including data dictionaries and onboarding playbooks; onboarded 3 analysts and cut ramp-up time from 3 weeks to under 10 days.
B5: Collaborated with product and engineering teams to define KPI frameworks; translated findings into executive dashboards adopted by 3 business units.

PROJECTS:
[Retail Demand Forecasting: ELT Pipeline] PySpark, Airflow, Snowflake, dbt, GCP, Docker, Apache Superset, Parquet | 2025
B1: Built end-to-end ELT pipeline on Snowflake ingesting 3 external APIs with dbt mart models and 5 Airflow DAGs; cut forecast generation time by 50% and pipeline failures by 80% through robust orchestration.
B2: Modeled time-series demand indices (YoY/MoM growth, seasonal demand, safety stock) across 20+ product categories; built Apache Superset dashboards across 10+ states, cutting report prep from 2 days to 30 min.

[Plate Planner: RAG Pipeline with Vector Storage] FastAPI, FAISS, Pinecone, Neo4j, SentenceTransformers, Docker | 2025
B1: Built data pipeline ingesting 223K+ recipes into FAISS vector storage enabling generative AI recommendations under 100ms query latency; exposed inference via clean REST API endpoints.
B2: Designed graph-based ingredient substitution engine in Neo4j with Word2Vec embeddings; achieved 70% Hit@5 accuracy and 20% quality gain, deployed end-to-end with Docker Compose.

[Inklude: Conditional Generative Model for Air-Drawing] PyTorch, CNN-BiLSTM, Pix2Pix GAN, Stable Diffusion, MediaPipe | 2024
B1: Developed hybrid CNN-BiLSTM-Attention classifier in PyTorch achieving 89.75% accuracy across 45 categories; introduced Motion Language Matrix representation enabling robust real-time sketch recognition.
B2: Trained 16 class-specific Pix2Pix generators on 16,000 curated image pairs using GAN + L1 + VGG perceptual loss; delivered complete real-time inference pipeline under 100ms latency.

[Retail Demand Forecasting & Inventory Planning] PySpark, MLlib, Random Forest, Streamlit, Docker | 2024
B1: Processed 1M+ retail transactions with PySpark MLlib; trained Random Forest regression (RMSE $421, MAE $174) and 3-class inventory classifier with 92% accuracy.
B2: Delivered interactive Streamlit dashboard with product forecast explorer, quarterly heatmaps, and holiday impact charts.

SKILLS:
Languages & Databases: Python, SQL, MySQL, Neo4j, REST APIs
Data Engineering: ETL/ELT Pipelines, PySpark, Apache Kafka, Airflow, dbt, Snowflake, Parquet, Data Modeling, Real-Time Streaming
Cloud & Infrastructure: GCP (Cloud Composer, Cloud Storage), Docker, FastAPI, Git, Linux
Visualization: Tableau, Power BI, Apache Superset, Streamlit
AI & ML: PyTorch, Scikit-learn, LLM Fine-Tuning, RAG Pipelines, FAISS, Pinecone, SentenceTransformers, Vector Databases
"""

# ── Jake's LaTeX template header ───────────────────────────────────────
LATEX_HEADER = r"""%-------------------------
% Resume in Latex — Jake's Template (github.com/jakegut)
%------------------------
\documentclass[letterpaper,10pt]{article}
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

\pagestyle{fancy}
\fancyhf{}
\fancyfoot{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

\addtolength{\oddsidemargin}{-0.6in}
\addtolength{\evensidemargin}{-0.6in}
\addtolength{\textwidth}{1.2in}
\addtolength{\topmargin}{-.65in}
\addtolength{\textheight}{1.3in}

\urlstyle{same}
\raggedbottom
\raggedright
\setlength{\tabcolsep}{0in}

\titleformat{\section}{
  \vspace{-8pt}\scshape\raggedright\large
}{}{0em}{}[\color{black}\titlerule \vspace{-7pt}]

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
\newcommand{\resumeSubHeadingListStart}{\begin{itemize}[leftmargin=0.15in, label={}]}
\newcommand{\resumeSubHeadingListEnd}{\end{itemize}}
\newcommand{\resumeItemListStart}{\begin{itemize}[leftmargin=0.2in, itemsep=0pt, topsep=2pt, parsep=0pt]}
\newcommand{\resumeItemListEnd}{\end{itemize}\vspace{-6pt}}

\begin{document}

\begin{center}
    \textbf{\Huge \scshape Shaila Reddy Kankanala} \\ \vspace{1pt}
    \small 925-819-3509 $|$
    \href{mailto:shailareddy1603@gmail.com}{\underline{shailareddy1603@gmail.com}} $|$
    \href{https://linkedin.com/in/shailareddy}{\underline{linkedin.com/in/shailareddy}} $|$
    San Jose, CA
\end{center}

\section{Education}
  \resumeSubHeadingListStart
    \resumeSubheading
      {San Jose State University}{Aug. 2024 -- May 2026}
      {Master of Science in Applied Data Intelligence}{San Jose, CA}
    \resumeSubheading
      {Jawaharlal Nehru Technological University}{Jul. 2018 -- Jul. 2022}
      {Bachelor of Technology in Information Technology}{Hyderabad, India}
  \resumeSubHeadingListEnd

"""


def escape_latex(s):
    """Escape special LaTeX characters."""
    replacements = [
        ('\\', r'\textbackslash{}'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('&', r'\&'),
        ('#', r'\#'),
        ('_', r'\_'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('~', r'\textasciitilde{}'),
        ('^', r'\textasciicircum{}'),
        ('<', r'\textless{}'),
        ('>', r'\textgreater{}'),
    ]
    # Don't double-escape backslashes from our own replacements
    s = s.replace('\\', '__BACKSLASH__')
    for char, repl in replacements[1:]:
        s = s.replace(char, repl)
    s = s.replace('__BACKSLASH__', r'\textbackslash{}')
    return s


def build_latex(data):
    tex = LATEX_HEADER

    # Experience
    tex += "\\section{Experience}\n  \\resumeSubHeadingListStart\n\n"
    for exp in data.get("experience", []):
        tex += f"    \\resumeSubheading\n"
        tex += f"      {{{exp['company']}}}{{{exp['dates']}}}\n"
        tex += f"      {{{exp['title']}}}{{{exp['location']}}}\n"
        tex += "      \\resumeItemListStart\n"
        for b in exp.get("bullets", []):
            tex += f"        \\resumeItem{{{escape_latex(b)}}}\n"
        tex += "      \\resumeItemListEnd\n\n"
    tex += "  \\resumeSubHeadingListEnd\n\n"

    # Projects
    tex += "\\section{Projects}\n    \\resumeSubHeadingListStart\n\n"
    for proj in data.get("projects", []):
        title = escape_latex(proj["title"])
        stack = escape_latex(proj["stack"])
        tex += f"      \\resumeProjectHeading\n"
        tex += f"          {{\\textbf{{{title}}} $|$ \\emph{{{stack}}}}}{{{proj['date']}}}\n"
        tex += "          \\resumeItemListStart\n"
        for b in proj.get("bullets", []):
            tex += f"            \\resumeItem{{{escape_latex(b)}}}\n"
        tex += "          \\resumeItemListEnd\n\n"
    tex += "    \\resumeSubHeadingListEnd\n\n"

    # Skills
    tex += "\\section{Technical Skills}\n"
    tex += " \\begin{itemize}[leftmargin=0.15in, label={}, itemsep=0pt, topsep=0pt]\n"
    tex += "    \\small{\\item{\n"
    lines = []
    for sk in data.get("skills", []):
        lines.append(f"     \\textbf{{{sk['category']}}}{{: {sk['items']}}}")
    tex += " \\\\\n".join(lines) + "\n"
    tex += "    }}\n \\end{itemize}\n\n\\end{document}\n"

    return tex


def compile_pdf(tex_content):
    """Compile LaTeX to PDF, return PDF bytes."""
    workdir = tempfile.mkdtemp()
    try:
        tex_path = os.path.join(workdir, "resume.tex")
        with open(tex_path, "w") as f:
            f.write(tex_content)

        for _ in range(2):  # compile twice for stable output
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "resume.tex"],
                cwd=workdir, capture_output=True, text=True, timeout=30
            )

        pdf_path = os.path.join(workdir, "resume.pdf")
        if not os.path.exists(pdf_path):
            raise RuntimeError("LaTeX compilation failed:\n" + result.stdout[-1000:])

        with open(pdf_path, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def make_prompt(jd):
    return f"""You are an expert ATS resume writer. Tailor Shaila's resume for this job description.

FIXED RULES:
- AllyIn.AI: exactly 5 bullets
- Dextara Datamatics: exactly 7 bullets
- RineX.AI: exactly 5 bullets
- Exactly 2 most relevant projects
- All dates/companies/titles exactly as in profile
- Rewrite bullets to mirror JD keywords — never invent facts
- STAR format, strong action verbs, no pronouns, tight sentences
- No special characters that break LaTeX: avoid %, $, &, #, _ unless you must
- Keep bullet text plain — no quotes, no parenthetical asides that use special chars

CANDIDATE PROFILE:
{PROFILE}

JOB DESCRIPTION:
{jd}

Return ONLY raw JSON, no markdown fences, no explanation:
{{"score":85,"role":"Role Title","fit":"One sentence fit assessment","matched_keywords":["k1","k2"],"missing_keywords":["k1"],"tips":"1. tip\\n2. tip\\n3. tip","experience":[{{"company":"AllyIn.AI","title":"AI Engineer Intern","dates":"May 2025 -- Aug. 2025","location":"San Jose, CA","bullets":["b1","b2","b3","b4","b5"]}},{{"company":"Dextara Datamatics","title":"Data Engineer","dates":"Aug. 2022 -- Jul. 2024","location":"Hyderabad, India","bullets":["b1","b2","b3","b4","b5","b6","b7"]}},{{"company":"RineX.AI","title":"Data Analyst Intern","dates":"Sept. 2021 -- Jul. 2022","location":"Hyderabad, India","bullets":["b1","b2","b3","b4","b5"]}}],"projects":[{{"title":"...","stack":"...","date":"2025","bullets":["b1","b2"]}},{{"title":"...","stack":"...","date":"2025","bullets":["b1","b2"]}}],"skills":[{{"category":"Languages & Databases","items":"..."}},{{"category":"AI & ML Frameworks","items":"..."}},{{"category":"Data Engineering","items":"..."}},{{"category":"Cloud & Infrastructure","items":"..."}},{{"category":"Visualization","items":"..."}}]}}"""


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/tailor", methods=["OPTIONS"])
def tailor_options():
    return "", 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/tailor", methods=["POST"])
def tailor():
    try:
        body = request.get_json()
        jd      = body.get("jd", "").strip()
        api_key = body.get("api_key", "").strip()

        if not jd:
            return jsonify({"error": "No job description provided"}), 400
        if not api_key:
            return jsonify({"error": "No API key provided"}), 400

        # Call Claude
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4000,
            messages=[{"role": "user", "content": make_prompt(jd)}]
        )

        raw = message.content[0].text
        # Clean JSON
        clean = raw.replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end   = clean.rfind("}")
        if start != -1 and end != -1:
            clean = clean[start:end+1]

        data = json.loads(clean)

        # Build and compile LaTeX
        tex = build_latex(data)
        pdf_bytes = compile_pdf(tex)

        # Return JSON with PDF as base64 + analysis data
        import base64
        return jsonify({
            "score":            data.get("score", 0),
            "role":             data.get("role", ""),
            "fit":              data.get("fit", ""),
            "matched_keywords": data.get("matched_keywords", []),
            "missing_keywords": data.get("missing_keywords", []),
            "tips":             data.get("tips", ""),
            "pdf_b64":          base64.b64encode(pdf_bytes).decode("utf-8")
        })

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Could not parse AI response: {str(e)}"}), 500
    except subprocess.TimeoutExpired:
        return jsonify({"error": "PDF compilation timed out"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
