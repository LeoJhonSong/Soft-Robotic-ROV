namespace SoftwareArmROV
{
    partial class frmPID
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(frmPID));
            this.panel1 = new System.Windows.Forms.Panel();
            this.btnSend = new System.Windows.Forms.Button();
            this.grpOption = new System.Windows.Forms.GroupBox();
            this.rdoStatic = new System.Windows.Forms.RadioButton();
            this.rdoRightShift = new System.Windows.Forms.RadioButton();
            this.rdoLeftShift = new System.Windows.Forms.RadioButton();
            this.rdoBackOff = new System.Windows.Forms.RadioButton();
            this.rdoForward = new System.Windows.Forms.RadioButton();
            this.rdoDepth = new System.Windows.Forms.RadioButton();
            this.grpPIDParam = new System.Windows.Forms.GroupBox();
            this.txtD = new System.Windows.Forms.TextBox();
            this.lblD = new System.Windows.Forms.Label();
            this.txtI = new System.Windows.Forms.TextBox();
            this.lblI = new System.Windows.Forms.Label();
            this.txtP = new System.Windows.Forms.TextBox();
            this.lblP = new System.Windows.Forms.Label();
            this.btnWrite = new System.Windows.Forms.Button();
            this.panel1.SuspendLayout();
            this.grpOption.SuspendLayout();
            this.grpPIDParam.SuspendLayout();
            this.SuspendLayout();
            // 
            // panel1
            // 
            this.panel1.Controls.Add(this.btnWrite);
            this.panel1.Controls.Add(this.btnSend);
            this.panel1.Controls.Add(this.grpOption);
            this.panel1.Controls.Add(this.grpPIDParam);
            this.panel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel1.Font = new System.Drawing.Font("宋体", 10.5F, System.Drawing.FontStyle.Bold);
            this.panel1.Location = new System.Drawing.Point(0, 0);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(439, 223);
            this.panel1.TabIndex = 0;
            // 
            // btnSend
            // 
            this.btnSend.Location = new System.Drawing.Point(232, 175);
            this.btnSend.Name = "btnSend";
            this.btnSend.Size = new System.Drawing.Size(75, 28);
            this.btnSend.TabIndex = 3;
            this.btnSend.Text = "发 送";
            this.btnSend.UseVisualStyleBackColor = true;
            this.btnSend.Click += new System.EventHandler(this.btnSend_Click);
            // 
            // grpOption
            // 
            this.grpOption.Controls.Add(this.rdoStatic);
            this.grpOption.Controls.Add(this.rdoRightShift);
            this.grpOption.Controls.Add(this.rdoLeftShift);
            this.grpOption.Controls.Add(this.rdoBackOff);
            this.grpOption.Controls.Add(this.rdoForward);
            this.grpOption.Controls.Add(this.rdoDepth);
            this.grpOption.Font = new System.Drawing.Font("宋体", 10.5F, System.Drawing.FontStyle.Bold);
            this.grpOption.Location = new System.Drawing.Point(12, 12);
            this.grpOption.Name = "grpOption";
            this.grpOption.Size = new System.Drawing.Size(415, 70);
            this.grpOption.TabIndex = 2;
            this.grpOption.TabStop = false;
            this.grpOption.Text = "PID选项";
            // 
            // rdoStatic
            // 
            this.rdoStatic.AutoSize = true;
            this.rdoStatic.Font = new System.Drawing.Font("宋体", 9F, System.Drawing.FontStyle.Bold);
            this.rdoStatic.Location = new System.Drawing.Point(327, 33);
            this.rdoStatic.Name = "rdoStatic";
            this.rdoStatic.Size = new System.Drawing.Size(49, 16);
            this.rdoStatic.TabIndex = 5;
            this.rdoStatic.TabStop = true;
            this.rdoStatic.Text = "静止";
            this.rdoStatic.UseVisualStyleBackColor = true;
            // 
            // rdoRightShift
            // 
            this.rdoRightShift.AutoSize = true;
            this.rdoRightShift.Font = new System.Drawing.Font("宋体", 9F, System.Drawing.FontStyle.Bold);
            this.rdoRightShift.Location = new System.Drawing.Point(272, 33);
            this.rdoRightShift.Name = "rdoRightShift";
            this.rdoRightShift.Size = new System.Drawing.Size(49, 16);
            this.rdoRightShift.TabIndex = 4;
            this.rdoRightShift.TabStop = true;
            this.rdoRightShift.Text = "右移";
            this.rdoRightShift.UseVisualStyleBackColor = true;
            // 
            // rdoLeftShift
            // 
            this.rdoLeftShift.AutoSize = true;
            this.rdoLeftShift.Font = new System.Drawing.Font("宋体", 9F, System.Drawing.FontStyle.Bold);
            this.rdoLeftShift.Location = new System.Drawing.Point(207, 33);
            this.rdoLeftShift.Name = "rdoLeftShift";
            this.rdoLeftShift.Size = new System.Drawing.Size(49, 16);
            this.rdoLeftShift.TabIndex = 3;
            this.rdoLeftShift.TabStop = true;
            this.rdoLeftShift.Text = "左移";
            this.rdoLeftShift.UseVisualStyleBackColor = true;
            // 
            // rdoBackOff
            // 
            this.rdoBackOff.AutoSize = true;
            this.rdoBackOff.Font = new System.Drawing.Font("宋体", 9F, System.Drawing.FontStyle.Bold);
            this.rdoBackOff.Location = new System.Drawing.Point(142, 33);
            this.rdoBackOff.Name = "rdoBackOff";
            this.rdoBackOff.Size = new System.Drawing.Size(49, 16);
            this.rdoBackOff.TabIndex = 2;
            this.rdoBackOff.TabStop = true;
            this.rdoBackOff.Text = "后退";
            this.rdoBackOff.UseVisualStyleBackColor = true;
            // 
            // rdoForward
            // 
            this.rdoForward.AutoSize = true;
            this.rdoForward.Font = new System.Drawing.Font("宋体", 9F, System.Drawing.FontStyle.Bold);
            this.rdoForward.Location = new System.Drawing.Point(77, 33);
            this.rdoForward.Name = "rdoForward";
            this.rdoForward.Size = new System.Drawing.Size(49, 16);
            this.rdoForward.TabIndex = 1;
            this.rdoForward.TabStop = true;
            this.rdoForward.Text = "前进";
            this.rdoForward.UseVisualStyleBackColor = true;
            // 
            // rdoDepth
            // 
            this.rdoDepth.AutoSize = true;
            this.rdoDepth.Checked = true;
            this.rdoDepth.Font = new System.Drawing.Font("宋体", 9F, System.Drawing.FontStyle.Bold);
            this.rdoDepth.Location = new System.Drawing.Point(12, 33);
            this.rdoDepth.Name = "rdoDepth";
            this.rdoDepth.Size = new System.Drawing.Size(49, 16);
            this.rdoDepth.TabIndex = 0;
            this.rdoDepth.TabStop = true;
            this.rdoDepth.Text = "深度";
            this.rdoDepth.UseVisualStyleBackColor = true;
            // 
            // grpPIDParam
            // 
            this.grpPIDParam.Controls.Add(this.txtD);
            this.grpPIDParam.Controls.Add(this.lblD);
            this.grpPIDParam.Controls.Add(this.txtI);
            this.grpPIDParam.Controls.Add(this.lblI);
            this.grpPIDParam.Controls.Add(this.txtP);
            this.grpPIDParam.Controls.Add(this.lblP);
            this.grpPIDParam.Font = new System.Drawing.Font("宋体", 10.5F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.grpPIDParam.Location = new System.Drawing.Point(12, 97);
            this.grpPIDParam.Name = "grpPIDParam";
            this.grpPIDParam.Size = new System.Drawing.Size(415, 72);
            this.grpPIDParam.TabIndex = 1;
            this.grpPIDParam.TabStop = false;
            this.grpPIDParam.Text = "PID参数";
            // 
            // txtD
            // 
            this.txtD.Location = new System.Drawing.Point(340, 32);
            this.txtD.Name = "txtD";
            this.txtD.Size = new System.Drawing.Size(48, 23);
            this.txtD.TabIndex = 5;
            // 
            // lblD
            // 
            this.lblD.AutoSize = true;
            this.lblD.Font = new System.Drawing.Font("宋体", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.lblD.Location = new System.Drawing.Point(299, 37);
            this.lblD.Name = "lblD";
            this.lblD.Size = new System.Drawing.Size(35, 12);
            this.lblD.TabIndex = 4;
            this.lblD.Text = "D值：";
            // 
            // txtI
            // 
            this.txtI.Location = new System.Drawing.Point(197, 32);
            this.txtI.Name = "txtI";
            this.txtI.Size = new System.Drawing.Size(48, 23);
            this.txtI.TabIndex = 3;
            // 
            // lblI
            // 
            this.lblI.AutoSize = true;
            this.lblI.Font = new System.Drawing.Font("宋体", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.lblI.Location = new System.Drawing.Point(156, 37);
            this.lblI.Name = "lblI";
            this.lblI.Size = new System.Drawing.Size(35, 12);
            this.lblI.TabIndex = 2;
            this.lblI.Text = "I值：";
            // 
            // txtP
            // 
            this.txtP.Location = new System.Drawing.Point(62, 32);
            this.txtP.Name = "txtP";
            this.txtP.Size = new System.Drawing.Size(48, 23);
            this.txtP.TabIndex = 1;
            // 
            // lblP
            // 
            this.lblP.AutoSize = true;
            this.lblP.Font = new System.Drawing.Font("宋体", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.lblP.Location = new System.Drawing.Point(21, 37);
            this.lblP.Name = "lblP";
            this.lblP.Size = new System.Drawing.Size(35, 12);
            this.lblP.TabIndex = 0;
            this.lblP.Text = "P值：";
            // 
            // btnWrite
            // 
            this.btnWrite.Location = new System.Drawing.Point(313, 176);
            this.btnWrite.Name = "btnWrite";
            this.btnWrite.Size = new System.Drawing.Size(114, 28);
            this.btnWrite.TabIndex = 4;
            this.btnWrite.Text = "写入FLASH";
            this.btnWrite.UseVisualStyleBackColor = true;
            this.btnWrite.Click += new System.EventHandler(this.btnWrite_Click);
            // 
            // frmPID
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(439, 223);
            this.Controls.Add(this.panel1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MaximizeBox = false;
            this.Name = "frmPID";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterParent;
            this.Text = "自控参数设置";
            this.panel1.ResumeLayout(false);
            this.grpOption.ResumeLayout(false);
            this.grpOption.PerformLayout();
            this.grpPIDParam.ResumeLayout(false);
            this.grpPIDParam.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.GroupBox grpOption;
        private System.Windows.Forms.GroupBox grpPIDParam;
        private System.Windows.Forms.TextBox txtD;
        private System.Windows.Forms.Label lblD;
        private System.Windows.Forms.TextBox txtI;
        private System.Windows.Forms.Label lblI;
        private System.Windows.Forms.TextBox txtP;
        private System.Windows.Forms.Label lblP;
        private System.Windows.Forms.RadioButton rdoRightShift;
        private System.Windows.Forms.RadioButton rdoLeftShift;
        private System.Windows.Forms.RadioButton rdoBackOff;
        private System.Windows.Forms.RadioButton rdoForward;
        private System.Windows.Forms.RadioButton rdoDepth;
        private System.Windows.Forms.Button btnSend;
        private System.Windows.Forms.RadioButton rdoStatic;
        private System.Windows.Forms.Button btnWrite;
    }
}