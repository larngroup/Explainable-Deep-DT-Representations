# Read csv dataset files
data_bubble<-read.csv("../data/davis/dataset/davis_dataset_processed.csv",header=TRUE)
protein_sequences<-data.frame(data_bubble$Sequence)
protein_sequences<-unique(protein_sequences)

library(Biostrings)
data(BLOSUM62)

# Smith-Waterman Local Algorithm
prot_sw_score <- function(sequence_a,sequence_b,sub_matrix,gap_opening,gap_extension,align_type){
  score_value<-pairwiseAlignment(AAString(sequence_a),
                                 AAString(sequence_b),
                                 substitutionMatrix=sub_matrix,
                                 gapOpening=gap_opening,
                                 gapExtension=gap_extension,
                                 type=align_type)
  
  return(score_value@score)
  
}

# Smith-Waterman Scores Normalized
prot_sw_score_normalized <- function(scores_matrix){
  scores_matrix_norm <- matrix(0,length(scores_matrix[,1]),length(scores_matrix[,1]))
  for (i in 1:length(scores_matrix[,1])){
    for (j in 1:length(scores_matrix[,1])) {
      
      scores_matrix_norm[i,j]<-scores_matrix[i,j]/(sqrt(scores_matrix[i,i]) * sqrt(scores_matrix[j,j]))
      
      
    }
  }
  return(scores_matrix_norm)
}

sequences<-protein_sequences
gap_opening<-10
gap_extension<-0.5
align_type<-'local'
sub_matrix<-BLOSUM62


scores_matrix <- matrix(0,length(sequences[,1]),length(sequences[,1]))
for (i in 1:length(sequences[,1])){
  for (j in 1:length(sequences[,1])){
    scores_matrix[i,j] <- prot_sw_score(sequences[i,1],sequences[j,1],sub_matrix,gap_opening,gap_extension,align_type)
  }
}

scores_values_norm_df <-as.data.frame(prot_sw_score_normalized(scores_matrix))
score_values_df <- as.data.frame(scores_matrix)

write.table(score_values_df,"../data/davis/similarity/protein_sw_score.csv",sep=",",row.names = sequences[,1], col.names = FALSE)
write.table(scores_values_norm_df,"../data/davis/similarity/protein_sw_score_norm.csv",sep=",",row.names = sequences[,1], col.names = FALSE)
