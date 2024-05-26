$(document).ready(function () {
    // Initialize Select2 on the descriptor and image_class select elements
    $('#descriptor').select2();
    $('#image_class').select2();

    $('#file').change(function () {
        var formData = new FormData();
        formData.append('file', $('#file')[0].files[0]);

        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (data) {
                if (data.error) {
                    alert(data.error);
                } else {
                    $('#preview').attr('src', data.file_path).removeClass('d-none');
                    $('#filename').val(data.filename);
                    $('#remove-image').removeClass('d-none');
                    $('button[type="submit"]').prop('disabled', false);
                }
            },
            error: function () {
                alert('Error uploading file.');
            }
        });
    });

    $('#remove-image').click(function (event) {
        event.preventDefault(); // Prevent default form submission
        event.stopPropagation(); // Stop event propagation

        var filename = $('#filename').val();
        $.ajax({
            url: '/delete/' + filename,
            type: 'POST',
            success: function (data) {
                if (data.error) {
                    alert(data.error);
                } else {
                    $('#preview').attr('src', '').addClass('d-none');
                    $('#filename').val('');
                    $('#remove-image').addClass('d-none');
                    $('#file').val('');
                    $('button[type="submit"]').prop('disabled', true);
                }
            },
            error: function () {
                alert('Error removing file.');
            }
        });
    });

    $('#search-form').submit(function (event) {
        event.preventDefault();
        var filename = $('#filename').val();
        if (!filename) {
            alert('Please upload an image first.');
            return;
        }

        // Clear previous results and RP curve
        $('#result-container').empty();
        $('#rp-curve').attr('src', '').addClass('d-none');

        $.ajax({
            url: '/search',
            type: 'POST',
            data: $(this).serialize(),
            success: function (data) {
                if (data.error) {
                    alert(data.error);
                } else {
                    $('#result-container').empty();
                    data.top20_similar_images.forEach(function (image) {
                        $('#result-container').append('<img src="' + image + '" class="result-img">');
                    });
                    if (data.rp_curve) {
                        var timestamp = new Date().getTime();
                        $('#rp-curve').attr('src', data.rp_curve + "?t=" + timestamp).removeClass('d-none');
                    }
                }
            },
            error: function () {
                alert('Error performing search.');
            }
        });
    });
});
