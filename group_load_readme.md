# Group Load - (Optimus OnYx 2.0)

> **Download this README:** Right-click on this content and select "Save As" to download as `README.md`

## Overview

Automated, scalable, and LLM-driven solution for health insurance **Group Load**: extraction, transformation, and loading of group and sub-group data as per client requirements.

## Project Description

This project enables automated extraction, transformation, and loading of **Groups** entities and their sub-structures within health insurance. It leverages state-of-the-art Large Language Models (LLMs) to parse client documentation, extract schema and business rule requirements, cluster logic into components, and generate the transformation code needed to process group and member details.

The system is designed for high reliability, repeatability, and ease of onboarding for new clients.

## Group Load Information

### Group Context in Health Insurance

A **Group** is an employer or organization that contracts with a health plan to provide insurance coverage to its employees or members. Groups link members to benefit plans, eligibility, and subgroups.

The Group is the **central organizing entity** for health insurance coverage, linking members to benefit plans, eligibility, and subgroups.

### Key Group Attributes

- **Group ID**: Unique identifier for each employer group
- **Group Name**: Name of the employer or organization
- **Effective/Termination Date**: Coverage time period for the group contract
- **Subgroups**: Divisions within a group, often by location, department, or employee class
- **Benefit Plans**: List and linkage of health insurance products/plans available to the group and its members

## Project Focus

This codebase is designed to **automate the Group Load process** for clients. It produces transformation scripts and data arrangements required by internal Operations team.

## Architecture & Workflow

### Step 1: LLM Scaffold (Code Generation)

**Purpose:**
- Reads TDDs and business rule PDFs
- Extracts all **Groups** and related fields
- Generates field/component mapping and Python transformation code

**Run:**
```bash
python llm_scaffold/llm_orchestrator.py
```

**Key Outputs:**
- Python transformation scripts for each grouped logic block
- Extract dependent cbms_tables for each component

### Step 2: Apply Generated Transformations

**Purpose:**
- Then generated code to transform source data into required Group Load files

**Run:**
```bash
python src/main.py
```

**Outputs:**
- Transformed data files
- Source/Format data cleaning
- Generalized cube and Response tables
- Target grouped cubes Column Health file categorizing of submission

**Communication Files:**
- Client data tables
- All-Final files

## Project Structure

```
llm_scaffold/
├── llm_orchestrator.py          # LLM pipeline controller
├── extract_target_fields.py
├── extract_rules.py
├── generate_code.py
├── transformation_components.py
└── templates/
    ├── create_components_pmt.py
    ├── gnrt_compnt_codes_pmt.py
    └── generate_codes_pmt.py

transformation_codes/
└── etc/

main.py                          # Data transformation runner
├── target_fields/
    └── CLIENT_ID_xxxx/
config/
└── config.py
```

## Environment Setup

### Prerequisites

Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

1. **Set Azure OpenAI environment variables:**
```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export DNOPOINT_URL="https://<your-endpoint>"
export DEPLOYMENT_NAME="gpt-4.1"
```

2. **Adjust configuration in `config/config.py` for:**
   - Paths to client documents (TDD, PDFs)
   - Input and output directories
   - Client/group identifiers

## Usage

### Example Usage

**Step 1: Scaffold Generation** (required on client or doc changes)
```bash
python llm_scaffold/llm_orchestrator.py
```

**Step 2: Apply Generated Transformations**
```bash
python src/main.py
```

**Order matters:** Always run scaffold before transformation if updating fields or rules!

## Troubleshooting & Best Practices

### Common Issues

**Transformation code/components missing:**
- Ensure LLM Scaffold ran successfully and latest docs/configs are in use
- Adding new group fields/components:
- Rerun the scaffold and then the main script
- Pipeline errors or incomplete outputs:
- Check for errors/logs in both scaffold and main script, and verify configuration files

## Technical Requirements

- Python 3.8+
- Azure OpenAI API access
- Required Python packages (see requirements.txt)
- Access to client TDD and business rule documentation

## Support

For technical support and additional information, please contact the internal Operations team or refer to the project documentation.