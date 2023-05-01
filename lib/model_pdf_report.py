#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime
from io import BytesIO

from PIL.Image import Image
# PDF processing
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors, units
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image,  Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER

def plot_to_image(plot, width, height):
    plot_memory = BytesIO()
    plot.savefig(plot_memory, format='png')
    plot_memory.seek(0)
    return Image(plot_memory, width=width, height=height)

def save_model_results_to_pdf(model_name, pr_curve, roc_curve, tpr_tnr_plot, conf_matrix, results, best_params):
    
    # Set up the PDF
    model_name_filename=model_name.replace(" ", "_")
    file_name = f"{model_name_filename}_Model_Summary.pdf"
    pdf = SimpleDocTemplate(file_name, 
                            pagesize=landscape(letter),
                            leftMargin=0.5 * units.inch,
                            rightMargin=0.5 * units.inch,
                            topMargin=0.5 * units.inch,
                            bottomMargin=0.5 * units.inch
    )

    # Set up the styles
    styles = getSampleStyleSheet()
    heading_style = styles['Heading1']
    date_style = styles['BodyText']

    # Create the heading
    heading = Paragraph(f'{model_name}', heading_style)

    # Create the date and time
    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_paragraph = Paragraph(date_time, date_style)
    
    # Create a table for the best hyperparameters
    hyperparameters_data = [['Parameter', 'Value']] + [[k, v] for k, v in best_params.items()]
    hyperparameters_table = Table(hyperparameters_data)
    
    # Apply styling to the hyperparameters table
    hyperparameters_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    # Convert the results DataFrame to a list including the header
    results_data = [results.columns.to_list()] + results.values.tolist()
    results_data = [
        [round(num, 3) if isinstance(num, (int, float)) else num for num in inner_list]
        for inner_list in results_data
    ]

    # Create a Table object from the DataFrame
    results_table = Table(results_data)

    # Apply styling to the table
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    # Generate images of each plot
    pr_curve_image = plot_to_image(pr_curve, width=300, height=200)
    roc_curve_image = plot_to_image(roc_curve, width=300, height=200)
    tpr_tnr_plot_image = plot_to_image(tpr_tnr_plot, width=300, height=200)
    conf_matrix_image = plot_to_image(conf_matrix, width=300, height=200)
    
    # Create a table with the four images
    image_table = Table([
        [pr_curve_image, roc_curve_image],
        [tpr_tnr_plot_image, conf_matrix_image]
    ], colWidths=[300, 300], rowHeights=[200, 200])
    
    # Style the image table (e.g., remove borders)
    image_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0, colors.transparent),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    
    # Create a list of content and add the elements to it
    content = []
    content.append(heading)
    content.append(Spacer(1, 2))
    content.append(date_paragraph)
    content.append(Spacer(1, 6))
    content.append(results_table)
    content.append(Spacer(1, 6))
    content.append(image_table)
    content.append(Spacer(1, 6))
    content.append(hyperparameters_table)

    # Build the PDF with the content list
    pdf.build(content)





# In[2]:


if __name__ == '__main__':
    print("Hello World")

