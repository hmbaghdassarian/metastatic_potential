library(tidyverse)
library(ggplot2)
library(ggpubr)
source('plotting_functions.R')
library(gridExtra)

### Load interaction analysis results----------------------
pairs_to_analyse <- c('PA2G4_SP1',
                      'C8A_SEMG1',
                      'ARRB2_SP1',
                      'NONO_AXIN1',
                      'APC_AXIN1',
                      'AXIN1_BRCA1',
                      'PCBP1_APC',
                      'AXIN1_PA2G4')
df <- data.frame()
plot_list <- NULL
i <- 1
for (pair in pairs_to_analyse){
  tmp <- data.table::fread(paste0('../synergy_analysis/synergy_analysis/',pair,'_grid_diff.csv')) %>%
    mutate(pairs=pair)
  colnames(tmp)[1:2] <- c('x','y')
  df <- rbind(df,tmp)
  p <- ggplot(tmp,
              aes(x=x,y=y,fill=mean))+
    geom_tile(color='#808080',size=0.2)+
    scale_fill_gradient2(midpoint = 0,low = '#467699',high = '#da3c48',mid='white')+
    labs(fill=expression(Delta('Metastatic potential')),
         x = paste0('stimulation of ',str_split_fixed(pair,'_',n=2)[1]),
         y=paste0('stimulation of ',str_split_fixed(pair,'_',n=2)[2]))+
    scale_x_continuous(n.breaks = 10)+
    scale_y_continuous(n.breaks = 10)
  p <- add_theme(p) +
    theme(legend.position = 'right',text = element_text(size=30),
          axis.title = element_text(size=32),
          panel.border = element_blank(),
          panel.grid = element_blank())
  p <- p+ theme(
    # Control ALL legend keys
    legend.key = element_rect(fill = "white", color = "black"),  # Key background
    legend.key.size = unit(1.5, "lines"),  # Key size (vertical units)
    
    # Text sizes
    legend.text = element_text(size = 20),  # Item labels
    legend.title = element_text(size = 24, face = "bold", hjust = 0.5),  # Titles
    
    # Spacing and arrangement
    legend.spacing = unit(0.25, "cm"),  # Space between legends
    legend.box.spacing = unit(0.5, "cm"), # Space around legend box
    legend.margin = margin(0.3, 0.3, 0.3, 0.3, "cm"),  # Outer margins
    
    # For color/fill legends specifically
    legend.key.width = unit(1.5, "cm"),  # Width of color keys
    legend.key.height = unit(1, "cm"),   # Height of color keys
    
    # For size legends
    legend.byrow = TRUE  # Arrange items in rows
  ) +
    
    # Additional control for continuous scales
    guides(
      color = guide_legend(
        override.aes = list(size = 4),  # Point size in legend
        nrow = 1  # Arrange in one row
      ),
      size = guide_legend(
        direction = "horizontal",
        title.position = "top"
      )
    )
  # print(p)
  ggsave(
    paste0('../synergy_analysis/heatmap',pair,'.eps'), 
    plot = p,
    device = cairo_ps,
    scale = 1,
    width = 12,
    height = 9,
    units = "in",
    dpi = 600,
  )
  plot_list[[i]] <- p
  i <- i+1
}

## Visualize in heatmaps-----------------------
final_plot <- ggarrange(plotlist=plot_list,
                   ncol=4,nrow=2,common.legend = F)
print(final_plot)
ggsave(
  '../synergy_analysis/top_heatmaps.png', 
  plot = final_plot,
  # device = cairo_ps,
  scale = 1,
  width = 24,
  height = 12,
  units = "in",
  dpi = 600,
)
