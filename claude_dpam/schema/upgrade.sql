-- Schema upgrade from version 1.0.0 to 1.1.0
-- Adds new functionality for domain analysis and reporting

-- First, record that we're starting the upgrade
INSERT INTO dpam_queue.schema_versions (version, description)
VALUES ('1.1.0', 'Schema upgrade with domain analysis and reporting features');

-- Add domain analysis tables
CREATE TABLE dpam_queue.domains (
    domain_id SERIAL PRIMARY KEY,
    structure_id INTEGER REFERENCES dpam_queue.structures(structure_id),
    domain_name VARCHAR(50),
    start_residue INTEGER,
    end_residue INTEGER,
    size INTEGER,
    confidence_level VARCHAR(20),
    confidence_score FLOAT,
    source VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW(),
    parameters JSONB
);

CREATE TABLE dpam_queue.domain_mappings (
    mapping_id SERIAL PRIMARY KEY,
    domain_id INTEGER REFERENCES dpam_queue.domains(domain_id),
    ecod_domain_id VARCHAR(50),
    mapping_method VARCHAR(20),
    confidence VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW(),
    parameters JSONB
);

CREATE TABLE dpam_queue.domain_secondary_structure (
    ss_id SERIAL PRIMARY KEY,
    domain_id INTEGER REFERENCES dpam_queue.domains(domain_id),
    helix_percent FLOAT,
    strand_percent FLOAT,
    loop_percent FLOAT,
    domain_class VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Add reporting tables
CREATE TABLE dpam_queue.reports (
    report_id SERIAL PRIMARY KEY,
    batch_id INTEGER REFERENCES dpam_queue.batches(batch_id),
    report_type VARCHAR(50),
    file_path TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    parameters JSONB
);

-- Add new columns to existing tables
ALTER TABLE dpam_queue.structures 
ADD COLUMN IF NOT EXISTS sequence TEXT,
ADD COLUMN IF NOT EXISTS method VARCHAR(50),
ADD COLUMN IF NOT EXISTS estimated_resolution FLOAT;

ALTER TABLE dpam_queue.batches
ADD COLUMN IF NOT EXISTS priority INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS tags JSONB,
ADD COLUMN IF NOT EXISTS last_modified_at TIMESTAMP;

-- Add tracking for step execution time
ALTER TABLE dpam_queue.step_logs
ADD COLUMN IF NOT EXISTS cpu_time FLOAT,
ADD COLUMN IF NOT EXISTS wall_time FLOAT,
ADD COLUMN IF NOT EXISTS memory_usage FLOAT;

-- Create new indexes for performance
CREATE INDEX IF NOT EXISTS idx_domains_structure_id ON dpam_queue.domains(structure_id);
CREATE INDEX IF NOT EXISTS idx_domain_mappings_domain_id ON dpam_queue.domain_mappings(domain_id);
CREATE INDEX IF NOT EXISTS idx_domain_mappings_ecod_domain_id ON dpam_queue.domain_mappings(ecod_domain_id);
CREATE INDEX IF NOT EXISTS idx_domain_secondary_structure_domain_id ON dpam_queue.domain_secondary_structure(domain_id);
CREATE INDEX IF NOT EXISTS idx_reports_batch_id ON dpam_queue.reports(batch_id);
CREATE INDEX IF NOT EXISTS idx_batches_priority ON dpam_queue.batches(priority);

-- Add useful views for reporting
CREATE OR REPLACE VIEW dpam_queue.batch_status_summary AS
SELECT 
    b.batch_id,
    b.name,
    b.status,
    COUNT(bi.batch_item_id) AS total_structures,
    SUM(CASE WHEN bi.status = 'COMPLETED' THEN 1 ELSE 0 END) AS completed_structures,
    SUM(CASE WHEN bi.status = 'FAILED' THEN 1 ELSE 0 END) AS failed_structures,
    SUM(CASE WHEN bi.status = 'PENDING' THEN 1 ELSE 0 END) AS pending_structures,
    COUNT(d.domain_id) AS total_domains,
    ROUND(COUNT(d.domain_id)::FLOAT / NULLIF(SUM(CASE WHEN bi.status = 'COMPLETED' THEN 1 ELSE 0 END), 0), 2) AS avg_domains_per_structure
FROM 
    dpam_queue.batches b
    LEFT JOIN dpam_queue.batch_items bi ON b.batch_id = bi.batch_id
    LEFT JOIN dpam_queue.structures s ON bi.structure_id = s.structure_id
    LEFT JOIN dpam_queue.domains d ON s.structure_id = d.structure_id
GROUP BY 
    b.batch_id, b.name, b.status;

CREATE OR REPLACE VIEW dpam_queue.domain_classification_summary AS
SELECT 
    dc.domain_class,
    COUNT(*) AS domain_count,
    AVG(d.size) AS avg_size,
    AVG(d.confidence_score) AS avg_confidence
FROM 
    dpam_queue.domains d
    JOIN dpam_queue.domain_secondary_structure dc ON d.domain_id = dc.domain_id
GROUP BY 
    dc.domain_class;

-- Update last_modified_at for existing batches
UPDATE dpam_queue.batches 
SET last_modified_at = GREATEST(created_at, completed_at);

-- Add trigger to maintain last_modified_at
CREATE OR REPLACE FUNCTION dpam_queue.update_batch_last_modified()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_modified_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_batch_last_modified_trigger
BEFORE UPDATE ON dpam_queue.batches
FOR EACH ROW EXECUTE FUNCTION dpam_queue.update_batch_last_modified();

-- Log the completion of the upgrade
UPDATE dpam_queue.schema_versions 
SET description = description || ' - Completed successfully'
WHERE version = '1.1.0';