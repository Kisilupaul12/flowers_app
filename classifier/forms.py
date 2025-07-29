# classifier/forms.py
from django import forms

class UploadImageForm(forms.Form):
    image = forms.ImageField(
        label='Select Flower Image',
        help_text='Upload a JPG, PNG, or other image file',
        widget=forms.FileInput(attrs={
            'accept': 'image/*',
            'class': 'form-control-file',
            'style': 'width: 100%; padding: 10px;'
        })
    )
    
    def clean_image(self):
        image = self.cleaned_data.get('image')
        
        if image:
            # Check file size (limit to 10MB)
            if image.size > 10 * 1024 * 1024:
                raise forms.ValidationError('Image file too large. Please upload an image smaller than 10MB.')
            
            # Check file extension
            allowed_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif']
            file_extension = image.name.split('.')[-1].lower()
            
            if file_extension not in allowed_extensions:
                raise forms.ValidationError(f'Please upload a valid image file. Allowed formats: {", ".join(allowed_extensions)}')
        
        return image
