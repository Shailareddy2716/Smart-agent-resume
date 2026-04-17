import os, json, base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

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
B2: Developed FastAPI-based APIs and Python libraries for ML researchers to manage and benchmark LLaMA fine-tuning outputs across 5 task categories, raising model accuracy from 72% to 90%.
B3: Shipped LLM evaluation and data quality tooling used daily by research teams; identified 3 critical dataset issues and lifted benchmark quality rating by 22% within 4 weeks.
B4: Consolidated 4 enterprise data sources into unified Pinecone vector store via Docker, boosting retrieval quality 30% and cutting deployment setup time by 60%.
B5: Prototyped sensor-to-inference event pipeline for emotion-aware AI agent, cutting architecture iteration cycles by 40% and accelerating time-to-demo by 2 weeks.

[Dextara Datamatics] Data Engineer | Aug. 2022 - Jul. 2024 | Hyderabad, India
B1: Engineered large-scale ETL/ELT pipelines in Python and SQL on Snowflake and MySQL processing 100K+ monthly transactions; consolidated CRM/ERP reporting across 5+ teams, cutting manual effort by 40%.
B2: Deployed Apache Kafka real-time streaming pipelines for high-volume event ingestion with fault tolerance, shrinking data freshness SLA from 4 hrs to under 30 min.
B3: Operated Apache Airflow pipelines via GCP Cloud Composer managing scheduling, retry logic, alerting, and lineage tracking in Docker; saved 8 hrs/week and improved observability across 20+ active DAGs.
B4: Authored dbt transformation models on Snowflake; built data cataloging and schema drift detection across 3 source systems, improving debugging time by 35% and preventing production incidents.
B5: Migrated 500K+ records integrating Salesforce, Acumatica, and Slack via REST APIs on GCP achieving 98% data consistency; mentored junior engineers on pipeline best practices.
B6: Embedded with US enterprise client as primary technical liaison; gathered requirements, built Tableau KPI dashboards across 5+ teams, and unblocked 3 delayed deliverables under delivery pressure.
B7: Designed data quality and provenance check frameworks across ingestion and transformation layers, reducing downstream reporting errors by 30% and improving SLA adherence across 4 business units.

[RineX.AI] Data Analyst Intern | Sept. 2021 - Jul. 2022 | Hyderabad, India
B1: Queried 1M+ row datasets via SQL and Python (Pandas, NumPy) to surface business KPIs; automated reporting workflows saving 12 hrs/week and cutting turnaround from 3 days to same-day.
B2: Identified data quality gaps across 6 source tables; built Python cleansing scripts cutting downstream report errors by 45% and saving 4 hrs of QA per cycle.
B3: Built Tableau and Power BI dashboards adopted across 4 teams; co-led churn analysis contributing to 15% retention uplift and monthly churn drop from 8% to 4.5% over 2 quarters.
B4: Developed dataset management tooling including data dictionaries and onboarding playbooks; onboarded 3 analysts and cut ramp-up time from 3 weeks to under 10 days.
B5: Collaborated with product and engineering teams to define KPI frameworks; translated findings into executive dashboards adopted by 3 business units.

PROJECTS:
[Retail Demand Forecasting: ELT Pipeline] PySpark, Airflow, Snowflake, dbt, GCP, Docker, Apache Superset, Parquet | 2025
B1: Built end-to-end ELT pipeline on Snowflake ingesting 3 external APIs with dbt mart models and 5 Airflow DAGs; cut forecast generation time by 50% and pipeline failures by 80%.
B2: Modeled time-series demand indices across 20+ product categories; built Apache Superset dashboards across 10+ states, cutting report prep from 2 days to 30 min.

[Plate Planner: RAG Pipeline with Vector Storage] FastAPI, FAISS, Pinecone, Neo4j, SentenceTransformers, Docker | 2025
B1: Built data pipeline ingesting 223K+ recipes into FAISS vector storage enabling generative AI recommendations under 100ms latency; exposed inference via clean REST API endpoints.
B2: Designed graph-based ingredient substitution engine in Neo4j with Word2Vec embeddings; achieved 70% Hit@5 accuracy and 20% quality gain, deployed end-to-end with Docker Compose.

[Inklude: Conditional Generative Model for Air-Drawing] PyTorch, CNN-BiLSTM, Pix2Pix GAN, Stable Diffusion, MediaPipe | 2024
B1: Developed hybrid CNN-BiLSTM-Attention classifier in PyTorch achieving 89.75% accuracy across 45 categories; introduced Motion Language Matrix representation enabling robust real-time sketch recognition.
B2: Trained 16 class-specific Pix2Pix generators on 16,000 curated image pairs; delivered complete real-time inference pipeline under 100ms latency.

SKILLS:
Languages & Databases: Python, SQL, MySQL, Neo4j, REST APIs
Data Engineering: ETL/ELT Pipelines, PySpark, Apache Kafka, Airflow, dbt, Snowflake, Parquet, Data Modeling, Real-Time Streaming
Cloud & Infrastructure: GCP (Cloud Composer, Cloud Storage), Docker, FastAPI, Git, Linux
Visualization: Tableau, Power BI, Apache Superset, Streamlit
AI & ML: PyTorch, Scikit-learn, LLM Fine-Tuning, RAG Pipelines, FAISS, Pinecone, SentenceTransformers, Vector Databases
"""

def make_prompt(jd):
    return f"""You are an expert ATS resume writer. Tailor Shaila's resume for this job description.
FIXED RULES: AllyIn.AI=5 bullets, Dextara=7 bullets, RineX=5 bullets, exactly 2 projects, all dates/titles exact, mirror JD keywords, STAR format, no pronouns, plain text only.
PROFILE:\n{PROFILE}\nJOB DESCRIPTION:\n{jd}
Return ONLY raw JSON no markdown:
{{"score":85,"role":"Title","fit":"sentence","matched_keywords":["k1"],"missing_keywords":["k2"],"tips":"1. tip\\n2. tip\\n3. tip","experience":[{{"company":"AllyIn.AI","title":"AI Engineer Intern","dates":"May 2025 - Aug. 2025","location":"San Jose, CA","bullets":["b1","b2","b3","b4","b5"]}},{{"company":"Dextara Datamatics","title":"Data Engineer","dates":"Aug. 2022 - Jul. 2024","location":"Hyderabad, India","bullets":["b1","b2","b3","b4","b5","b6","b7"]}},{{"company":"RineX.AI","title":"Data Analyst Intern","dates":"Sept. 2021 - Jul. 2022","location":"Hyderabad, India","bullets":["b1","b2","b3","b4","b5"]}}],"projects":[{{"title":"...","stack":"...","date":"2025","bullets":["b1","b2"]}},{{"title":"...","stack":"...","date":"2025","bullets":["b1","b2"]}}],"skills":[{{"category":"Languages & Databases","items":"..."}},{{"category":"AI & ML Frameworks","items":"..."}},{{"category":"Data Engineering","items":"..."}},{{"category":"Cloud & Infrastructure","items":"..."}},{{"category":"Visualization","items":"..."}}]}}"""

PW, PH = letter
LM = RM = 43.2
TW = PW - LM - RM

def wrap(c, text, font, size, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if c.stringWidth(t, font, size) <= max_w: cur = t
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines or [""]

def build_pdf(data):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    y = PH - 28

    def rstr(txt, yy, font="Helvetica", size=9):
        c.setFont(font, size); c.setFillColorRGB(0.25,0.25,0.25)
        c.drawString(PW-RM-c.stringWidth(txt,font,size), yy, txt)
        c.setFillColorRGB(0,0,0)

    def rule():
        nonlocal y
        c.setLineWidth(0.4); c.line(LM, y, PW-RM, y); y -= 6

    def sp(h):
        nonlocal y; y -= h

    def chk(n=14):
        nonlocal y
        if y-n < 24: c.showPage(); y = PH-28

    def bullet_block(bullets):
        nonlocal y
        c.setFont("Helvetica", 9)
        for b in bullets:
            lines = wrap(c, b, "Helvetica", 9, TW-15)
            for i, ln in enumerate(lines):
                chk(12)
                if i == 0:
                    c.setFillColorRGB(0,0,0)
                    c.rect(LM+2, y-5, 3.2, 3.2, fill=1, stroke=0)
                    c.drawString(LM+13, y, ln)
                else:
                    c.drawString(LM+16, y, ln)
                y -= 11.2
            y -= 0.8

    def sec(title):
        nonlocal y
        chk(20)
        c.setFont("Helvetica-Bold", 11.5)
        c.drawString(LM, y, title.upper()); sp(4); rule()

    # Name
    c.setFont("Helvetica-Bold", 15)
    nw = c.stringWidth("Shaila Reddy Kankanala","Helvetica-Bold",15)
    c.drawString((PW-nw)/2, y, "Shaila Reddy Kankanala"); sp(17)
    c.setFont("Helvetica", 9); c.setFillColorRGB(0.2,0.2,0.2)
    ct = "925-819-3509  |  shailareddy1603@gmail.com  |  linkedin.com/in/shailareddy  |  San Jose, CA"
    c.drawString((PW-c.stringWidth(ct,"Helvetica",9))/2, y, ct)
    c.setFillColorRGB(0,0,0); sp(13)

    # Education
    sec("Education")
    for inst, dates, deg, loc in [
        ("San Jose State University","Aug. 2024 - May 2026","Master of Science in Applied Data Intelligence","San Jose, CA"),
        ("Jawaharlal Nehru Technological University","Jul. 2018 - Jul. 2022","Bachelor of Technology in Information Technology","Hyderabad, India"),
    ]:
        chk(22)
        c.setFont("Helvetica-Bold",9.5); c.drawString(LM,y,inst); rstr(dates,y); sp(12)
        c.setFont("Helvetica-Oblique",9.5); c.drawString(LM+10,y,deg); rstr(loc,y,"Helvetica-Oblique"); sp(13)

    # Experience
    sec("Experience")
    for exp in data.get("experience",[]):
        chk(26)
        c.setFont("Helvetica-Bold",9.5); c.drawString(LM,y,exp["company"]); rstr(exp["dates"],y); sp(12)
        c.setFont("Helvetica-Oblique",9.5); c.drawString(LM+10,y,exp["title"]); rstr(exp["location"],y,"Helvetica-Oblique"); sp(11)
        bullet_block(exp.get("bullets",[])); sp(4)

    # Projects
    sec("Projects")
    for proj in data.get("projects",[]):
        chk(22)
        c.setFont("Helvetica-Bold",9.5)
        pipe = proj["title"] + "  |  "; pw2 = c.stringWidth(pipe,"Helvetica-Bold",9.5)
        c.drawString(LM, y, pipe)
        dw = c.stringWidth(proj["date"],"Helvetica",9)+4
        stk = proj.get("stack","")
        c.setFont("Helvetica-Oblique",9)
        while c.stringWidth(stk,"Helvetica-Oblique",9) > TW-pw2-dw and ", " in stk:
            stk = stk.rsplit(", ",1)[0]
        c.drawString(LM+pw2, y, stk); rstr(proj["date"],y); sp(11)
        bullet_block(proj.get("bullets",[])); sp(4)

    # Skills
    sec("Technical Skills")
    for sk in data.get("skills",[]):
        chk(13)
        c.setFont("Helvetica-Bold",9)
        cat = sk["category"]+": "; cw2 = c.stringWidth(cat,"Helvetica-Bold",9)
        c.drawString(LM,y,cat); c.setFont("Helvetica",9)
        ils = wrap(c, sk.get("items",""), "Helvetica", 9, TW-cw2)
        c.drawString(LM+cw2, y, ils[0])
        for ln in ils[1:]: y-=11.2; chk(12); c.drawString(LM+cw2,y,ln)
        sp(12)

    c.save(); buf.seek(0)
    return buf.read()


@app.route("/health", methods=["GET"])
def health(): return jsonify({"status":"ok"})

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
        if s!=-1 and e!=-1: clean = clean[s:e+1]
        data = json.loads(clean)

        pdf_bytes = build_pdf(data)
        return jsonify({
            "score":data.get("score",0),"role":data.get("role",""),
            "fit":data.get("fit",""),"matched_keywords":data.get("matched_keywords",[]),
            "missing_keywords":data.get("missing_keywords",[]),"tips":data.get("tips",""),
            "pdf_b64":base64.b64encode(pdf_bytes).decode("utf-8")
        })
    except json.JSONDecodeError as e:
        return jsonify({"error":f"Could not parse AI response: {e}. Try again."}),500
    except Exception as e:
        return jsonify({"error":str(e)}),500

if __name__=="__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
