## init.sql

CREATE SCHEMA dpam_queue

-- Proteins table
CREATE TABLE dpam_queue.proteins (
    protein_id SERIAL PRIMARY KEY,
    uniprot_id VARCHAR(10),
    name TEXT,
    description TEXT,
    sequence TEXT,
    length INTEGER
);

-- Structures table
CREATE TABLE dpam_queue.structures (
    structure_id SERIAL PRIMARY KEY,
    protein_id INTEGER REFERENCES proteins(protein_id),
    pdb_id VARCHAR(10),
    resolution FLOAT,
    structure_path TEXT,
    format VARCHAR(10),
    processing_date TIMESTAMP,
    processing_status VARCHAR(20)
);

-- Batches table
CREATE TABLE dpam_queue.batches (
    batch_id SERIAL PRIMARY KEY,
    name TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    status VARCHAR(20),
    parameters JSONB
);

-- Batch items table
CREATE TABLE dpam_queue.batch_items (
    batch_item_id SERIAL PRIMARY KEY,
    batch_id INTEGER REFERENCES batches(batch_id),
    structure_id INTEGER REFERENCES structures(structure_id),
    status VARCHAR(20),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    results JSONB
);

-- Pipeline steps table
CREATE TABLE dpam_queue.pipeline_steps (
    step_id SERIAL PRIMARY KEY,
    name TEXT,
    description TEXT,
    script_path TEXT,
    order_index INTEGER
);

-- Step logs table
CREATE TABLE dpam_queue.step_logs (
    log_id SERIAL PRIMARY KEY,
    batch_item_id INTEGER REFERENCES batch_items(batch_item_id),
    step_id INTEGER REFERENCES pipeline_steps(step_id),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR(20),
    output TEXT,
    metrics JSONB
);

-- Batch checkpoints table
CREATE TABLE dpam_queue.batch_checkpoints (
    checkpoint_id SERIAL PRIMARY KEY,
    batch_id INTEGER REFERENCES dpam_queue.batches(batch_id),
    step_name VARCHAR(50),
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Grid jobs table
CREATE TABLE dpam_queue.grid_jobs (
    grid_job_id SERIAL PRIMARY KEY,
    batch_id INTEGER REFERENCES dpam_queue.batches(batch_id),
    step_name VARCHAR(50),
    job_id VARCHAR(50),
    array_size INTEGER,
    status VARCHAR(20),
    task_stats JSONB,
    submitted_at TIMESTAMP,
    updated_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- Error logs table
CREATE TABLE dpam_queue.error_logs (
    error_id SERIAL PRIMARY KEY,
    batch_id INTEGER REFERENCES dpam_queue.batches(batch_id),
    structure_id INTEGER REFERENCES dpam_queue.structures(structure_id),
    step_name VARCHAR(50),
    error_message TEXT,
    error_type VARCHAR(50),
    occurred_at TIMESTAMP DEFAULT NOW(),
    resolved_at TIMESTAMP,
    resolution_strategy VARCHAR(50)
);

-- Indices for batch operations
CREATE INDEX idx_batch_items_batch_id ON dpam_queue.batch_items(batch_id);
CREATE INDEX idx_batch_items_structure_id ON dpam_queue.batch_items(structure_id);
CREATE INDEX idx_step_logs_batch_item_id ON dpam_queue.step_logs(batch_item_id);
CREATE INDEX idx_step_logs_step_id ON dpam_queue.step_logs(step_id);
CREATE INDEX idx_batch_checkpoints_batch_id ON dpam_queue.batch_checkpoints(batch_id);
CREATE INDEX idx_grid_jobs_batch_id ON dpam_queue.grid_jobs(batch_id);
CREATE INDEX idx_error_logs_batch_id ON dpam_queue.error_logs(batch_id);
CREATE INDEX idx_error_logs_structure_id ON dpam_queue.error_logs(structure_id);

-- Indices for structure operations
CREATE INDEX idx_structures_protein_id ON dpam_queue.structures(protein_id);
CREATE INDEX idx_structures_pdb_id ON dpam_queue.structures(pdb_id);
CREATE INDEX idx_proteins_uniprot_id ON dpam_queue.proteins(uniprot_id);

-- Schema version tracking
CREATE TABLE dpam_queue.schema_versions (
    version_id SERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL,
    applied_at TIMESTAMP DEFAULT NOW(),
    description TEXT
);

-- Insert initial version
INSERT INTO dpam_queue.schema_versions (version, description)
VALUES ('1.0.0', 'Initial schema creation');