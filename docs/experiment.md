# Experimental validation of unZipro

unZipro was validated across 9 diverse protein engineering tasks, achieving consistent improvements in functional assays:

|      **Category**      | **Name** | **PDB  ID** | **Length** | **Resolution/  Confidence** | **Success  rate** | Objective                          |
| :--------------------: | :------: | :---------: | :--------: | :-------------------------: | :---------------: | ---------------------------------- |
|       Deaminase        |  TadA8e  |    6VPC     |  167  aa   |           3.2  Å            |       66.7%       | Improved base-editing efficiency   |
|        Nuclease        |  SpCas9  |    4OO8     |  1368  aa  |           2.5  Å            |        45%        | Enhanced genome-editing efficiency |
|        Nuclease        |  CasΦ2   |    7LYS     |  756  aa   |           3.05  Å           |       37.5%       | Enhanced genome-editing efficiency |
|       Polymerase       | MMLV-RT  |  8WUT,8WUV  |  496  aa   |             3 Å             |       78.3%       | Enhanced prime-editing efficiency  |
|      Exonuclease       |   T5E    | AlphaFold3  |  291  aa   |         pLDDT=0.93          |       100%        | Enhanced genome-editing efficiency |
|       Luciferase       |   LUC    |    1LCI     |  550  aa   |           2.5  Å            |       57.1%       | Increased fluorescence             |
| Transcription  factors |  OsPHR2  | AlphaFold3  |   426 aa   |         pLDDT=0.43          |       50.0%       | Enhanced transcriptional activity  |
| Transcription  factors |  OsNAC3  | AlphaFold3  |  276  aa   |         pLDDT=0.69          |       70.0%       | Enhanced transcriptional activity  |
|   Antiviral  protein   |  HvMS1   | AlphaFold3  |  766  aa   |         pLDDT=0.93          |       61.1%       | Reduced pathogen virulence         |


> unZipro achieves up to 28× improvement in desired protein properties,
and the success rate of high-fitness variants (>1.1× WT) reaches up to 100% (average 61%).

Each script reproduces the mutation prioritization results for a specific protein category.
The results include ranked residue-wise mutation probabilities, and predicted high-fitness variants.
All scripts can be executed directly after environment setup (see [Installation](https://github.com/Gabriel-QIN/unZipro/blob/master/runs/install_unZipro.sh))

💡 Tip:
You can modify the --pdb, --param, and --outdir flags in each script to test your own proteins or finetuned models.

| Category               | Script                   | Description                                             |
| ---------------------- | ------------------------ | ------------------------------------------------------- |
| **Genome editors**     | `runs/run_ABE.sh`        | Adenine base editor (TadA8e)                            |
|                        | `runs/run_nuclease.sh`   | Three nucleases (SpCas9, CasΦ2/Cas12j2, T5E)            |
|                        | `runs/run_polymerase.sh` | MMLV reverse transcriptase under multiple conformations |
| **Fluorescent enzyme** | `runs/run_luciferase.sh` | Luciferase for improved fluorescence intensity          |
| **Plant proteins**     | `runs/run_plantTF.sh`    | DNA-binding domains of plant transcription factors      |
|                        | `runs/run_R_protein.sh`  | Plant virus-resistance (R) proteins     