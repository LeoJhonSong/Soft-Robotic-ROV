namespace SoftwareArmROV
{
    partial class frmControl
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
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(frmControl));
            this.panel1 = new System.Windows.Forms.Panel();
            this.ssrStatusInfor = new System.Windows.Forms.StatusStrip();
            this.toolStripStatusLabel1 = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolSStLbl = new System.Windows.Forms.ToolStripStatusLabel();
            this.ttrType = new System.Windows.Forms.ToolStripStatusLabel();
            this.tssJoyCon = new System.Windows.Forms.ToolStripStatusLabel();
            this.tssConTerminal = new System.Windows.Forms.ToolStripStatusLabel();
            this.tssTime = new System.Windows.Forms.ToolStripStatusLabel();
            this.grpOperation = new System.Windows.Forms.GroupBox();
            this.cmbHandle = new System.Windows.Forms.ComboBox();
            this.btnConHandle = new System.Windows.Forms.Button();
            this.lblHandle = new System.Windows.Forms.Label();
            this.grpLEDAndSteeringEngine = new System.Windows.Forms.GroupBox();
            this.btnClose_Steering_Engine2 = new System.Windows.Forms.Button();
            this.trbSteering_Engine2 = new System.Windows.Forms.TrackBar();
            this.lblSteering_Engine2 = new System.Windows.Forms.Label();
            this.btnClose_Steering_Engine1 = new System.Windows.Forms.Button();
            this.trbSteering_Engine1 = new System.Windows.Forms.TrackBar();
            this.lblSteering_Engine1 = new System.Windows.Forms.Label();
            this.btnCloseLED = new System.Windows.Forms.Button();
            this.trbLED = new System.Windows.Forms.TrackBar();
            this.lblLED = new System.Windows.Forms.Label();
            this.grpType = new System.Windows.Forms.GroupBox();
            this.rdoCloseLoop = new System.Windows.Forms.RadioButton();
            this.rdoOpenLoop = new System.Windows.Forms.RadioButton();
            this.grpTest = new System.Windows.Forms.GroupBox();
            this.btnCloseTranslation = new System.Windows.Forms.Button();
            this.trbTranslation = new System.Windows.Forms.TrackBar();
            this.lblTranslation = new System.Windows.Forms.Label();
            this.btnCloseRotate = new System.Windows.Forms.Button();
            this.trbRotate = new System.Windows.Forms.TrackBar();
            this.lblRotate = new System.Windows.Forms.Label();
            this.btnCloseAround = new System.Windows.Forms.Button();
            this.trbAround = new System.Windows.Forms.TrackBar();
            this.lblAround = new System.Windows.Forms.Label();
            this.btnCloseVertical = new System.Windows.Forms.Button();
            this.trbVertical = new System.Windows.Forms.TrackBar();
            this.lblVertical = new System.Windows.Forms.Label();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.btnConSocket = new System.Windows.Forms.Button();
            this.txtPort = new System.Windows.Forms.TextBox();
            this.lblPort = new System.Windows.Forms.Label();
            this.txtIP = new System.Windows.Forms.TextBox();
            this.lbltIP = new System.Windows.Forms.Label();
            this.grpRockerStateShow = new System.Windows.Forms.GroupBox();
            this.lblTurnRight = new System.Windows.Forms.Label();
            this.lblTurnLeft = new System.Windows.Forms.Label();
            this.hscZ = new System.Windows.Forms.HScrollBar();
            this.vscR = new System.Windows.Forms.VScrollBar();
            this.lblBackOff = new System.Windows.Forms.Label();
            this.lblForward = new System.Windows.Forms.Label();
            this.lblRightShift = new System.Windows.Forms.Label();
            this.lblLeftShift = new System.Windows.Forms.Label();
            this.hscX = new System.Windows.Forms.HScrollBar();
            this.vscY = new System.Windows.Forms.VScrollBar();
            this.lblDive = new System.Windows.Forms.Label();
            this.lblFloating = new System.Windows.Forms.Label();
            this.grpMachineStateShow = new System.Windows.Forms.GroupBox();
            this.label6 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.hiIndicator = new HeadingIndicator.HeadingIndicator();
            this.lbltMainDeckCabin = new System.Windows.Forms.Label();
            this.picMainDeckCabin = new System.Windows.Forms.PictureBox();
            this.lbltControlModule = new System.Windows.Forms.Label();
            this.picControlModule = new System.Windows.Forms.PictureBox();
            this.lblElectricity2 = new System.Windows.Forms.Label();
            this.lbltElectricity2 = new System.Windows.Forms.Label();
            this.lblVoltage2 = new System.Windows.Forms.Label();
            this.lbltVoltage2 = new System.Windows.Forms.Label();
            this.lblElectricity1 = new System.Windows.Forms.Label();
            this.lbltElectricity1 = new System.Windows.Forms.Label();
            this.lblVoltage1 = new System.Windows.Forms.Label();
            this.lbltVoltage1 = new System.Windows.Forms.Label();
            this.lblDepth = new System.Windows.Forms.Label();
            this.lbltDepth = new System.Windows.Forms.Label();
            this.lblBoard = new System.Windows.Forms.Label();
            this.lblTBoard = new System.Windows.Forms.Label();
            this.lblTemperature = new System.Windows.Forms.Label();
            this.lblttemperature = new System.Windows.Forms.Label();
            this.skE = new Sunisoft.IrisSkin.SkinEngine(((System.ComponentModel.Component)(this)));
            this.tmrHandle = new System.Windows.Forms.Timer(this.components);
            this.tmrUI = new System.Windows.Forms.Timer(this.components);
            this.tmrSys = new System.Windows.Forms.Timer(this.components);
            this.panel1.SuspendLayout();
            this.ssrStatusInfor.SuspendLayout();
            this.grpOperation.SuspendLayout();
            this.grpLEDAndSteeringEngine.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trbSteering_Engine2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.trbSteering_Engine1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.trbLED)).BeginInit();
            this.grpType.SuspendLayout();
            this.grpTest.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trbTranslation)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.trbRotate)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.trbAround)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.trbVertical)).BeginInit();
            this.groupBox1.SuspendLayout();
            this.grpRockerStateShow.SuspendLayout();
            this.grpMachineStateShow.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.picMainDeckCabin)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.picControlModule)).BeginInit();
            this.SuspendLayout();
            // 
            // panel1
            // 
            this.panel1.BackColor = System.Drawing.SystemColors.Control;
            this.panel1.Controls.Add(this.ssrStatusInfor);
            this.panel1.Controls.Add(this.grpOperation);
            this.panel1.Controls.Add(this.grpLEDAndSteeringEngine);
            this.panel1.Controls.Add(this.grpType);
            this.panel1.Controls.Add(this.grpTest);
            this.panel1.Controls.Add(this.groupBox1);
            this.panel1.Controls.Add(this.grpRockerStateShow);
            this.panel1.Controls.Add(this.grpMachineStateShow);
            this.panel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel1.Location = new System.Drawing.Point(0, 0);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(1031, 703);
            this.panel1.TabIndex = 0;
            // 
            // ssrStatusInfor
            // 
            this.ssrStatusInfor.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripStatusLabel1,
            this.toolSStLbl,
            this.ttrType,
            this.tssJoyCon,
            this.tssConTerminal,
            this.tssTime});
            this.ssrStatusInfor.Location = new System.Drawing.Point(0, 677);
            this.ssrStatusInfor.Name = "ssrStatusInfor";
            this.ssrStatusInfor.Size = new System.Drawing.Size(1031, 26);
            this.ssrStatusInfor.TabIndex = 8;
            this.ssrStatusInfor.Text = "statusStrip1";
            // 
            // toolStripStatusLabel1
            // 
            this.toolStripStatusLabel1.Font = new System.Drawing.Font("微软雅黑", 12F);
            this.toolStripStatusLabel1.ForeColor = System.Drawing.SystemColors.Window;
            this.toolStripStatusLabel1.Name = "toolStripStatusLabel1";
            this.toolStripStatusLabel1.Size = new System.Drawing.Size(90, 21);
            this.toolStripStatusLabel1.Text = "状态显示：";
            // 
            // toolSStLbl
            // 
            this.toolSStLbl.Font = new System.Drawing.Font("微软雅黑", 12F);
            this.toolSStLbl.ForeColor = System.Drawing.SystemColors.Window;
            this.toolSStLbl.Name = "toolSStLbl";
            this.toolSStLbl.Size = new System.Drawing.Size(121, 21);
            this.toolSStLbl.Text = "服务器未创建   ";
            // 
            // ttrType
            // 
            this.ttrType.Font = new System.Drawing.Font("微软雅黑", 12F);
            this.ttrType.ForeColor = System.Drawing.SystemColors.Window;
            this.ttrType.Name = "ttrType";
            this.ttrType.Size = new System.Drawing.Size(0, 21);
            // 
            // tssJoyCon
            // 
            this.tssJoyCon.Font = new System.Drawing.Font("微软雅黑", 12F);
            this.tssJoyCon.ForeColor = System.Drawing.SystemColors.Window;
            this.tssJoyCon.Name = "tssJoyCon";
            this.tssJoyCon.Size = new System.Drawing.Size(0, 21);
            // 
            // tssConTerminal
            // 
            this.tssConTerminal.Font = new System.Drawing.Font("微软雅黑", 12F);
            this.tssConTerminal.ForeColor = System.Drawing.SystemColors.Window;
            this.tssConTerminal.Name = "tssConTerminal";
            this.tssConTerminal.Size = new System.Drawing.Size(0, 21);
            // 
            // tssTime
            // 
            this.tssTime.Font = new System.Drawing.Font("微软雅黑", 12F);
            this.tssTime.ForeColor = System.Drawing.SystemColors.Window;
            this.tssTime.Name = "tssTime";
            this.tssTime.Size = new System.Drawing.Size(0, 21);
            // 
            // grpOperation
            // 
            this.grpOperation.Controls.Add(this.cmbHandle);
            this.grpOperation.Controls.Add(this.btnConHandle);
            this.grpOperation.Controls.Add(this.lblHandle);
            this.grpOperation.Enabled = false;
            this.grpOperation.Font = new System.Drawing.Font("宋体", 13F);
            this.grpOperation.Location = new System.Drawing.Point(616, 197);
            this.grpOperation.Name = "grpOperation";
            this.grpOperation.Size = new System.Drawing.Size(398, 70);
            this.grpOperation.TabIndex = 6;
            this.grpOperation.TabStop = false;
            this.grpOperation.Text = "操作类型";
            // 
            // cmbHandle
            // 
            this.cmbHandle.FormattingEnabled = true;
            this.cmbHandle.Location = new System.Drawing.Point(89, 30);
            this.cmbHandle.Name = "cmbHandle";
            this.cmbHandle.Size = new System.Drawing.Size(188, 25);
            this.cmbHandle.TabIndex = 27;
            this.cmbHandle.SelectedIndexChanged += new System.EventHandler(this.cmbHandle_SelectedIndexChanged);
            // 
            // btnConHandle
            // 
            this.btnConHandle.Font = new System.Drawing.Font("宋体", 10.5F, System.Drawing.FontStyle.Bold);
            this.btnConHandle.Location = new System.Drawing.Point(295, 28);
            this.btnConHandle.Name = "btnConHandle";
            this.btnConHandle.Size = new System.Drawing.Size(80, 30);
            this.btnConHandle.TabIndex = 26;
            this.btnConHandle.Text = "连 接";
            this.btnConHandle.UseVisualStyleBackColor = true;
            this.btnConHandle.Click += new System.EventHandler(this.btnConHandle_Click);
            // 
            // lblHandle
            // 
            this.lblHandle.AutoSize = true;
            this.lblHandle.Font = new System.Drawing.Font("宋体", 14F);
            this.lblHandle.Location = new System.Drawing.Point(13, 36);
            this.lblHandle.Name = "lblHandle";
            this.lblHandle.Size = new System.Drawing.Size(85, 19);
            this.lblHandle.TabIndex = 25;
            this.lblHandle.Text = "请选择：";
            // 
            // grpLEDAndSteeringEngine
            // 
            this.grpLEDAndSteeringEngine.Controls.Add(this.btnClose_Steering_Engine2);
            this.grpLEDAndSteeringEngine.Controls.Add(this.trbSteering_Engine2);
            this.grpLEDAndSteeringEngine.Controls.Add(this.lblSteering_Engine2);
            this.grpLEDAndSteeringEngine.Controls.Add(this.btnClose_Steering_Engine1);
            this.grpLEDAndSteeringEngine.Controls.Add(this.trbSteering_Engine1);
            this.grpLEDAndSteeringEngine.Controls.Add(this.lblSteering_Engine1);
            this.grpLEDAndSteeringEngine.Controls.Add(this.btnCloseLED);
            this.grpLEDAndSteeringEngine.Controls.Add(this.trbLED);
            this.grpLEDAndSteeringEngine.Controls.Add(this.lblLED);
            this.grpLEDAndSteeringEngine.Font = new System.Drawing.Font("宋体", 13F);
            this.grpLEDAndSteeringEngine.Location = new System.Drawing.Point(616, 281);
            this.grpLEDAndSteeringEngine.Name = "grpLEDAndSteeringEngine";
            this.grpLEDAndSteeringEngine.Size = new System.Drawing.Size(398, 152);
            this.grpLEDAndSteeringEngine.TabIndex = 5;
            this.grpLEDAndSteeringEngine.TabStop = false;
            this.grpLEDAndSteeringEngine.Text = "照明及舵机操作";
            // 
            // btnClose_Steering_Engine2
            // 
            this.btnClose_Steering_Engine2.Font = new System.Drawing.Font("宋体", 10.5F, System.Drawing.FontStyle.Bold);
            this.btnClose_Steering_Engine2.Location = new System.Drawing.Point(296, 110);
            this.btnClose_Steering_Engine2.Name = "btnClose_Steering_Engine2";
            this.btnClose_Steering_Engine2.Size = new System.Drawing.Size(80, 30);
            this.btnClose_Steering_Engine2.TabIndex = 60;
            this.btnClose_Steering_Engine2.Text = "关 闭";
            this.btnClose_Steering_Engine2.UseVisualStyleBackColor = true;
            this.btnClose_Steering_Engine2.Click += new System.EventHandler(this.btnClose_Steering_Engine2_Click);
            // 
            // trbSteering_Engine2
            // 
            this.trbSteering_Engine2.AutoSize = false;
            this.trbSteering_Engine2.Location = new System.Drawing.Point(78, 116);
            this.trbSteering_Engine2.Maximum = 2000;
            this.trbSteering_Engine2.Name = "trbSteering_Engine2";
            this.trbSteering_Engine2.Size = new System.Drawing.Size(205, 30);
            this.trbSteering_Engine2.TabIndex = 59;
            // 
            // lblSteering_Engine2
            // 
            this.lblSteering_Engine2.AutoSize = true;
            this.lblSteering_Engine2.Font = new System.Drawing.Font("宋体", 14F);
            this.lblSteering_Engine2.Location = new System.Drawing.Point(16, 119);
            this.lblSteering_Engine2.Name = "lblSteering_Engine2";
            this.lblSteering_Engine2.Size = new System.Drawing.Size(57, 19);
            this.lblSteering_Engine2.TabIndex = 58;
            this.lblSteering_Engine2.Text = "舵机2";
            // 
            // btnClose_Steering_Engine1
            // 
            this.btnClose_Steering_Engine1.Font = new System.Drawing.Font("宋体", 10.5F, System.Drawing.FontStyle.Bold);
            this.btnClose_Steering_Engine1.Location = new System.Drawing.Point(296, 67);
            this.btnClose_Steering_Engine1.Name = "btnClose_Steering_Engine1";
            this.btnClose_Steering_Engine1.Size = new System.Drawing.Size(80, 30);
            this.btnClose_Steering_Engine1.TabIndex = 57;
            this.btnClose_Steering_Engine1.Text = "关 闭";
            this.btnClose_Steering_Engine1.UseVisualStyleBackColor = true;
            this.btnClose_Steering_Engine1.Click += new System.EventHandler(this.btnClose_Steering_Engine1_Click);
            // 
            // trbSteering_Engine1
            // 
            this.trbSteering_Engine1.AutoSize = false;
            this.trbSteering_Engine1.Location = new System.Drawing.Point(78, 68);
            this.trbSteering_Engine1.Maximum = 2000;
            this.trbSteering_Engine1.Name = "trbSteering_Engine1";
            this.trbSteering_Engine1.Size = new System.Drawing.Size(205, 30);
            this.trbSteering_Engine1.TabIndex = 56;
            // 
            // lblSteering_Engine1
            // 
            this.lblSteering_Engine1.AutoSize = true;
            this.lblSteering_Engine1.Font = new System.Drawing.Font("宋体", 14F);
            this.lblSteering_Engine1.Location = new System.Drawing.Point(16, 73);
            this.lblSteering_Engine1.Name = "lblSteering_Engine1";
            this.lblSteering_Engine1.Size = new System.Drawing.Size(57, 19);
            this.lblSteering_Engine1.TabIndex = 55;
            this.lblSteering_Engine1.Text = "舵机1";
            // 
            // btnCloseLED
            // 
            this.btnCloseLED.Font = new System.Drawing.Font("宋体", 10.5F, System.Drawing.FontStyle.Bold);
            this.btnCloseLED.Location = new System.Drawing.Point(296, 24);
            this.btnCloseLED.Name = "btnCloseLED";
            this.btnCloseLED.Size = new System.Drawing.Size(80, 30);
            this.btnCloseLED.TabIndex = 54;
            this.btnCloseLED.Text = "关 闭";
            this.btnCloseLED.UseVisualStyleBackColor = true;
            this.btnCloseLED.Click += new System.EventHandler(this.btnCloseLED_Click);
            // 
            // trbLED
            // 
            this.trbLED.AutoSize = false;
            this.trbLED.Location = new System.Drawing.Point(78, 25);
            this.trbLED.Maximum = 950;
            this.trbLED.Name = "trbLED";
            this.trbLED.Size = new System.Drawing.Size(205, 30);
            this.trbLED.TabIndex = 53;
            // 
            // lblLED
            // 
            this.lblLED.AutoSize = true;
            this.lblLED.Font = new System.Drawing.Font("宋体", 14F);
            this.lblLED.Location = new System.Drawing.Point(19, 27);
            this.lblLED.Name = "lblLED";
            this.lblLED.Size = new System.Drawing.Size(47, 19);
            this.lblLED.TabIndex = 52;
            this.lblLED.Text = "照明";
            // 
            // grpType
            // 
            this.grpType.Controls.Add(this.rdoCloseLoop);
            this.grpType.Controls.Add(this.rdoOpenLoop);
            this.grpType.Font = new System.Drawing.Font("宋体", 13F);
            this.grpType.Location = new System.Drawing.Point(616, 124);
            this.grpType.Name = "grpType";
            this.grpType.Size = new System.Drawing.Size(398, 58);
            this.grpType.TabIndex = 4;
            this.grpType.TabStop = false;
            this.grpType.Text = "控制模式";
            // 
            // rdoCloseLoop
            // 
            this.rdoCloseLoop.AutoSize = true;
            this.rdoCloseLoop.Checked = true;
            this.rdoCloseLoop.Font = new System.Drawing.Font("宋体", 14F);
            this.rdoCloseLoop.Location = new System.Drawing.Point(240, 25);
            this.rdoCloseLoop.Name = "rdoCloseLoop";
            this.rdoCloseLoop.Size = new System.Drawing.Size(65, 23);
            this.rdoCloseLoop.TabIndex = 1;
            this.rdoCloseLoop.TabStop = true;
            this.rdoCloseLoop.Text = "闭环";
            this.rdoCloseLoop.UseVisualStyleBackColor = true;
            this.rdoCloseLoop.Click += new System.EventHandler(this.rdoCloseLoop_Click);
            // 
            // rdoOpenLoop
            // 
            this.rdoOpenLoop.AutoSize = true;
            this.rdoOpenLoop.Font = new System.Drawing.Font("宋体", 14F);
            this.rdoOpenLoop.Location = new System.Drawing.Point(89, 25);
            this.rdoOpenLoop.Name = "rdoOpenLoop";
            this.rdoOpenLoop.Size = new System.Drawing.Size(65, 23);
            this.rdoOpenLoop.TabIndex = 0;
            this.rdoOpenLoop.Text = "开环";
            this.rdoOpenLoop.UseVisualStyleBackColor = true;
            this.rdoOpenLoop.Click += new System.EventHandler(this.rdoOpenLoop_Click);
            // 
            // grpTest
            // 
            this.grpTest.Controls.Add(this.btnCloseTranslation);
            this.grpTest.Controls.Add(this.trbTranslation);
            this.grpTest.Controls.Add(this.lblTranslation);
            this.grpTest.Controls.Add(this.btnCloseRotate);
            this.grpTest.Controls.Add(this.trbRotate);
            this.grpTest.Controls.Add(this.lblRotate);
            this.grpTest.Controls.Add(this.btnCloseAround);
            this.grpTest.Controls.Add(this.trbAround);
            this.grpTest.Controls.Add(this.lblAround);
            this.grpTest.Controls.Add(this.btnCloseVertical);
            this.grpTest.Controls.Add(this.trbVertical);
            this.grpTest.Controls.Add(this.lblVertical);
            this.grpTest.Font = new System.Drawing.Font("宋体", 13F);
            this.grpTest.Location = new System.Drawing.Point(616, 445);
            this.grpTest.Name = "grpTest";
            this.grpTest.Size = new System.Drawing.Size(398, 212);
            this.grpTest.TabIndex = 3;
            this.grpTest.TabStop = false;
            this.grpTest.Text = "电机测试";
            // 
            // btnCloseTranslation
            // 
            this.btnCloseTranslation.Font = new System.Drawing.Font("宋体", 10.5F, System.Drawing.FontStyle.Bold);
            this.btnCloseTranslation.Location = new System.Drawing.Point(295, 76);
            this.btnCloseTranslation.Name = "btnCloseTranslation";
            this.btnCloseTranslation.Size = new System.Drawing.Size(80, 30);
            this.btnCloseTranslation.TabIndex = 57;
            this.btnCloseTranslation.Text = "关 闭";
            this.btnCloseTranslation.UseVisualStyleBackColor = true;
            this.btnCloseTranslation.Click += new System.EventHandler(this.btnCloseTranslation_Click);
            // 
            // trbTranslation
            // 
            this.trbTranslation.AutoSize = false;
            this.trbTranslation.Location = new System.Drawing.Point(75, 76);
            this.trbTranslation.Maximum = 255;
            this.trbTranslation.Name = "trbTranslation";
            this.trbTranslation.Size = new System.Drawing.Size(205, 30);
            this.trbTranslation.TabIndex = 56;
            this.trbTranslation.Scroll += new System.EventHandler(this.trbTranslation_Scroll);
            // 
            // lblTranslation
            // 
            this.lblTranslation.AutoSize = true;
            this.lblTranslation.Font = new System.Drawing.Font("宋体", 14F);
            this.lblTranslation.Location = new System.Drawing.Point(18, 82);
            this.lblTranslation.Name = "lblTranslation";
            this.lblTranslation.Size = new System.Drawing.Size(47, 19);
            this.lblTranslation.TabIndex = 55;
            this.lblTranslation.Text = "平移";
            // 
            // btnCloseRotate
            // 
            this.btnCloseRotate.Font = new System.Drawing.Font("宋体", 10.5F, System.Drawing.FontStyle.Bold);
            this.btnCloseRotate.Location = new System.Drawing.Point(295, 166);
            this.btnCloseRotate.Name = "btnCloseRotate";
            this.btnCloseRotate.Size = new System.Drawing.Size(80, 30);
            this.btnCloseRotate.TabIndex = 51;
            this.btnCloseRotate.Text = "关 闭";
            this.btnCloseRotate.UseVisualStyleBackColor = true;
            this.btnCloseRotate.Click += new System.EventHandler(this.btnCloseRotate_Click);
            // 
            // trbRotate
            // 
            this.trbRotate.AutoSize = false;
            this.trbRotate.Location = new System.Drawing.Point(75, 169);
            this.trbRotate.Maximum = 255;
            this.trbRotate.Name = "trbRotate";
            this.trbRotate.Size = new System.Drawing.Size(205, 25);
            this.trbRotate.TabIndex = 50;
            this.trbRotate.Scroll += new System.EventHandler(this.trbRotate_Scroll);
            // 
            // lblRotate
            // 
            this.lblRotate.AutoSize = true;
            this.lblRotate.Font = new System.Drawing.Font("宋体", 14F);
            this.lblRotate.Location = new System.Drawing.Point(18, 172);
            this.lblRotate.Name = "lblRotate";
            this.lblRotate.Size = new System.Drawing.Size(47, 19);
            this.lblRotate.TabIndex = 49;
            this.lblRotate.Text = "旋转";
            // 
            // btnCloseAround
            // 
            this.btnCloseAround.Font = new System.Drawing.Font("宋体", 10.5F, System.Drawing.FontStyle.Bold);
            this.btnCloseAround.Location = new System.Drawing.Point(296, 121);
            this.btnCloseAround.Name = "btnCloseAround";
            this.btnCloseAround.Size = new System.Drawing.Size(80, 30);
            this.btnCloseAround.TabIndex = 48;
            this.btnCloseAround.Text = "关 闭";
            this.btnCloseAround.UseVisualStyleBackColor = true;
            this.btnCloseAround.Click += new System.EventHandler(this.btnCloseAround_Click);
            // 
            // trbAround
            // 
            this.trbAround.AutoSize = false;
            this.trbAround.Location = new System.Drawing.Point(76, 124);
            this.trbAround.Maximum = 255;
            this.trbAround.Name = "trbAround";
            this.trbAround.Size = new System.Drawing.Size(207, 25);
            this.trbAround.TabIndex = 47;
            this.trbAround.Scroll += new System.EventHandler(this.trbAround_Scroll);
            // 
            // lblAround
            // 
            this.lblAround.AutoSize = true;
            this.lblAround.Font = new System.Drawing.Font("宋体", 14F);
            this.lblAround.Location = new System.Drawing.Point(19, 127);
            this.lblAround.Name = "lblAround";
            this.lblAround.Size = new System.Drawing.Size(47, 19);
            this.lblAround.TabIndex = 46;
            this.lblAround.Text = "前后";
            // 
            // btnCloseVertical
            // 
            this.btnCloseVertical.Font = new System.Drawing.Font("宋体", 10.5F, System.Drawing.FontStyle.Bold);
            this.btnCloseVertical.Location = new System.Drawing.Point(294, 31);
            this.btnCloseVertical.Name = "btnCloseVertical";
            this.btnCloseVertical.Size = new System.Drawing.Size(80, 30);
            this.btnCloseVertical.TabIndex = 45;
            this.btnCloseVertical.Text = "关 闭";
            this.btnCloseVertical.UseVisualStyleBackColor = true;
            this.btnCloseVertical.Click += new System.EventHandler(this.btnCloseVertical_Click);
            // 
            // trbVertical
            // 
            this.trbVertical.AutoSize = false;
            this.trbVertical.Location = new System.Drawing.Point(77, 31);
            this.trbVertical.Maximum = 255;
            this.trbVertical.Name = "trbVertical";
            this.trbVertical.Size = new System.Drawing.Size(205, 30);
            this.trbVertical.TabIndex = 44;
            // 
            // lblVertical
            // 
            this.lblVertical.AutoSize = true;
            this.lblVertical.Font = new System.Drawing.Font("宋体", 14F);
            this.lblVertical.Location = new System.Drawing.Point(18, 37);
            this.lblVertical.Name = "lblVertical";
            this.lblVertical.Size = new System.Drawing.Size(47, 19);
            this.lblVertical.TabIndex = 43;
            this.lblVertical.Text = "垂直";
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.btnConSocket);
            this.groupBox1.Controls.Add(this.txtPort);
            this.groupBox1.Controls.Add(this.lblPort);
            this.groupBox1.Controls.Add(this.txtIP);
            this.groupBox1.Controls.Add(this.lbltIP);
            this.groupBox1.Font = new System.Drawing.Font("宋体", 13F);
            this.groupBox1.Location = new System.Drawing.Point(616, 12);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(398, 97);
            this.groupBox1.TabIndex = 2;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "通信设置";
            // 
            // btnConSocket
            // 
            this.btnConSocket.Font = new System.Drawing.Font("宋体", 13F, System.Drawing.FontStyle.Bold);
            this.btnConSocket.Location = new System.Drawing.Point(301, 22);
            this.btnConSocket.Name = "btnConSocket";
            this.btnConSocket.Size = new System.Drawing.Size(80, 68);
            this.btnConSocket.TabIndex = 22;
            this.btnConSocket.Text = "创建服务器";
            this.btnConSocket.UseVisualStyleBackColor = true;
            this.btnConSocket.Click += new System.EventHandler(this.btnConSocket_Click);
            // 
            // txtPort
            // 
            this.txtPort.Location = new System.Drawing.Point(75, 64);
            this.txtPort.Name = "txtPort";
            this.txtPort.Size = new System.Drawing.Size(210, 27);
            this.txtPort.TabIndex = 16;
            this.txtPort.Text = "9090";
            // 
            // lblPort
            // 
            this.lblPort.AutoSize = true;
            this.lblPort.Font = new System.Drawing.Font("宋体", 14F);
            this.lblPort.Location = new System.Drawing.Point(6, 65);
            this.lblPort.Name = "lblPort";
            this.lblPort.Size = new System.Drawing.Size(57, 19);
            this.lblPort.TabIndex = 15;
            this.lblPort.Text = " 端口";
            // 
            // txtIP
            // 
            this.txtIP.Location = new System.Drawing.Point(75, 24);
            this.txtIP.Name = "txtIP";
            this.txtIP.Size = new System.Drawing.Size(210, 27);
            this.txtIP.TabIndex = 14;
            this.txtIP.Text = "192.168.1.112";
            // 
            // lbltIP
            // 
            this.lbltIP.Font = new System.Drawing.Font("宋体", 14F);
            this.lbltIP.Location = new System.Drawing.Point(26, 25);
            this.lbltIP.Name = "lbltIP";
            this.lbltIP.Size = new System.Drawing.Size(30, 20);
            this.lbltIP.TabIndex = 13;
            this.lbltIP.Text = "IP";
            // 
            // grpRockerStateShow
            // 
            this.grpRockerStateShow.Controls.Add(this.lblTurnRight);
            this.grpRockerStateShow.Controls.Add(this.lblTurnLeft);
            this.grpRockerStateShow.Controls.Add(this.hscZ);
            this.grpRockerStateShow.Controls.Add(this.vscR);
            this.grpRockerStateShow.Controls.Add(this.lblBackOff);
            this.grpRockerStateShow.Controls.Add(this.lblForward);
            this.grpRockerStateShow.Controls.Add(this.lblRightShift);
            this.grpRockerStateShow.Controls.Add(this.lblLeftShift);
            this.grpRockerStateShow.Controls.Add(this.hscX);
            this.grpRockerStateShow.Controls.Add(this.vscY);
            this.grpRockerStateShow.Controls.Add(this.lblDive);
            this.grpRockerStateShow.Controls.Add(this.lblFloating);
            this.grpRockerStateShow.Font = new System.Drawing.Font("宋体", 13F);
            this.grpRockerStateShow.Location = new System.Drawing.Point(12, 328);
            this.grpRockerStateShow.Name = "grpRockerStateShow";
            this.grpRockerStateShow.Size = new System.Drawing.Size(583, 329);
            this.grpRockerStateShow.TabIndex = 1;
            this.grpRockerStateShow.TabStop = false;
            this.grpRockerStateShow.Text = "摇杆状态显示";
            // 
            // lblTurnRight
            // 
            this.lblTurnRight.AutoSize = true;
            this.lblTurnRight.Font = new System.Drawing.Font("宋体", 14F);
            this.lblTurnRight.Location = new System.Drawing.Point(509, 294);
            this.lblTurnRight.Name = "lblTurnRight";
            this.lblTurnRight.Size = new System.Drawing.Size(47, 19);
            this.lblTurnRight.TabIndex = 26;
            this.lblTurnRight.Text = "右转";
            // 
            // lblTurnLeft
            // 
            this.lblTurnLeft.AutoSize = true;
            this.lblTurnLeft.Font = new System.Drawing.Font("宋体", 14F);
            this.lblTurnLeft.Location = new System.Drawing.Point(288, 293);
            this.lblTurnLeft.Name = "lblTurnLeft";
            this.lblTurnLeft.Size = new System.Drawing.Size(47, 19);
            this.lblTurnLeft.TabIndex = 25;
            this.lblTurnLeft.Text = "左转";
            // 
            // hscZ
            // 
            this.hscZ.Enabled = false;
            this.hscZ.Location = new System.Drawing.Point(335, 293);
            this.hscZ.Maximum = 255;
            this.hscZ.Name = "hscZ";
            this.hscZ.Size = new System.Drawing.Size(170, 20);
            this.hscZ.TabIndex = 24;
            // 
            // vscR
            // 
            this.vscR.Enabled = false;
            this.vscR.Location = new System.Drawing.Point(413, 69);
            this.vscR.Maximum = 255;
            this.vscR.Name = "vscR";
            this.vscR.Size = new System.Drawing.Size(20, 190);
            this.vscR.TabIndex = 23;
            // 
            // lblBackOff
            // 
            this.lblBackOff.AutoSize = true;
            this.lblBackOff.Font = new System.Drawing.Font("宋体", 14F);
            this.lblBackOff.Location = new System.Drawing.Point(398, 264);
            this.lblBackOff.Name = "lblBackOff";
            this.lblBackOff.Size = new System.Drawing.Size(47, 19);
            this.lblBackOff.TabIndex = 20;
            this.lblBackOff.Text = "后退";
            // 
            // lblForward
            // 
            this.lblForward.AutoSize = true;
            this.lblForward.Font = new System.Drawing.Font("宋体", 14F);
            this.lblForward.Location = new System.Drawing.Point(397, 41);
            this.lblForward.Name = "lblForward";
            this.lblForward.Size = new System.Drawing.Size(47, 19);
            this.lblForward.TabIndex = 19;
            this.lblForward.Text = "前进";
            // 
            // lblRightShift
            // 
            this.lblRightShift.AutoSize = true;
            this.lblRightShift.Font = new System.Drawing.Font("宋体", 14F);
            this.lblRightShift.Location = new System.Drawing.Point(240, 294);
            this.lblRightShift.Name = "lblRightShift";
            this.lblRightShift.Size = new System.Drawing.Size(47, 19);
            this.lblRightShift.TabIndex = 18;
            this.lblRightShift.Text = "右移";
            // 
            // lblLeftShift
            // 
            this.lblLeftShift.AutoSize = true;
            this.lblLeftShift.Font = new System.Drawing.Font("宋体", 14F);
            this.lblLeftShift.Location = new System.Drawing.Point(10, 293);
            this.lblLeftShift.Name = "lblLeftShift";
            this.lblLeftShift.Size = new System.Drawing.Size(47, 19);
            this.lblLeftShift.TabIndex = 17;
            this.lblLeftShift.Text = "左移";
            // 
            // hscX
            // 
            this.hscX.Enabled = false;
            this.hscX.Location = new System.Drawing.Point(67, 293);
            this.hscX.Maximum = 255;
            this.hscX.Name = "hscX";
            this.hscX.Size = new System.Drawing.Size(170, 20);
            this.hscX.TabIndex = 16;
            // 
            // vscY
            // 
            this.vscY.Enabled = false;
            this.vscY.Location = new System.Drawing.Point(146, 69);
            this.vscY.Maximum = 255;
            this.vscY.Name = "vscY";
            this.vscY.Size = new System.Drawing.Size(20, 190);
            this.vscY.TabIndex = 15;
            // 
            // lblDive
            // 
            this.lblDive.AutoSize = true;
            this.lblDive.Font = new System.Drawing.Font("宋体", 14F);
            this.lblDive.Location = new System.Drawing.Point(133, 264);
            this.lblDive.Name = "lblDive";
            this.lblDive.Size = new System.Drawing.Size(47, 19);
            this.lblDive.TabIndex = 12;
            this.lblDive.Text = "下潜";
            // 
            // lblFloating
            // 
            this.lblFloating.AutoSize = true;
            this.lblFloating.Font = new System.Drawing.Font("宋体", 14F);
            this.lblFloating.Location = new System.Drawing.Point(134, 41);
            this.lblFloating.Name = "lblFloating";
            this.lblFloating.Size = new System.Drawing.Size(47, 19);
            this.lblFloating.TabIndex = 11;
            this.lblFloating.Text = "上浮";
            // 
            // grpMachineStateShow
            // 
            this.grpMachineStateShow.Controls.Add(this.label6);
            this.grpMachineStateShow.Controls.Add(this.label7);
            this.grpMachineStateShow.Controls.Add(this.label5);
            this.grpMachineStateShow.Controls.Add(this.label4);
            this.grpMachineStateShow.Controls.Add(this.label3);
            this.grpMachineStateShow.Controls.Add(this.label2);
            this.grpMachineStateShow.Controls.Add(this.label1);
            this.grpMachineStateShow.Controls.Add(this.hiIndicator);
            this.grpMachineStateShow.Controls.Add(this.lbltMainDeckCabin);
            this.grpMachineStateShow.Controls.Add(this.picMainDeckCabin);
            this.grpMachineStateShow.Controls.Add(this.lbltControlModule);
            this.grpMachineStateShow.Controls.Add(this.picControlModule);
            this.grpMachineStateShow.Controls.Add(this.lblElectricity2);
            this.grpMachineStateShow.Controls.Add(this.lbltElectricity2);
            this.grpMachineStateShow.Controls.Add(this.lblVoltage2);
            this.grpMachineStateShow.Controls.Add(this.lbltVoltage2);
            this.grpMachineStateShow.Controls.Add(this.lblElectricity1);
            this.grpMachineStateShow.Controls.Add(this.lbltElectricity1);
            this.grpMachineStateShow.Controls.Add(this.lblVoltage1);
            this.grpMachineStateShow.Controls.Add(this.lbltVoltage1);
            this.grpMachineStateShow.Controls.Add(this.lblDepth);
            this.grpMachineStateShow.Controls.Add(this.lbltDepth);
            this.grpMachineStateShow.Controls.Add(this.lblBoard);
            this.grpMachineStateShow.Controls.Add(this.lblTBoard);
            this.grpMachineStateShow.Controls.Add(this.lblTemperature);
            this.grpMachineStateShow.Controls.Add(this.lblttemperature);
            this.grpMachineStateShow.Font = new System.Drawing.Font("宋体", 13F);
            this.grpMachineStateShow.Location = new System.Drawing.Point(12, 12);
            this.grpMachineStateShow.Name = "grpMachineStateShow";
            this.grpMachineStateShow.Size = new System.Drawing.Size(583, 299);
            this.grpMachineStateShow.TabIndex = 0;
            this.grpMachineStateShow.TabStop = false;
            this.grpMachineStateShow.Text = "机器人状态显示";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(222, 258);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(17, 18);
            this.label6.TabIndex = 25;
            this.label6.Text = "A";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(222, 220);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(17, 18);
            this.label7.TabIndex = 24;
            this.label7.Text = "V";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(222, 182);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(17, 18);
            this.label5.TabIndex = 23;
            this.label5.Text = "A";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(222, 144);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(17, 18);
            this.label4.TabIndex = 22;
            this.label4.Text = "V";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(222, 106);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(26, 18);
            this.label3.TabIndex = 21;
            this.label3.Text = "米";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(222, 70);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(35, 18);
            this.label2.TabIndex = 20;
            this.label2.Text = "°C";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(222, 35);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(35, 18);
            this.label1.TabIndex = 19;
            this.label1.Text = "°C";
            // 
            // hiIndicator
            // 
            this.hiIndicator.ControlWidth = 150;
            this.hiIndicator.CurrentHeading = new decimal(new int[] {
            0,
            0,
            0,
            65536});
            this.hiIndicator.Location = new System.Drawing.Point(351, 130);
            this.hiIndicator.Name = "hiIndicator";
            this.hiIndicator.Size = new System.Drawing.Size(150, 150);
            this.hiIndicator.TabIndex = 18;
            this.hiIndicator.Text = "headingIndicator1";
            // 
            // lbltMainDeckCabin
            // 
            this.lbltMainDeckCabin.AutoSize = true;
            this.lbltMainDeckCabin.Font = new System.Drawing.Font("宋体", 14F);
            this.lbltMainDeckCabin.Location = new System.Drawing.Point(329, 90);
            this.lbltMainDeckCabin.Name = "lbltMainDeckCabin";
            this.lbltMainDeckCabin.Size = new System.Drawing.Size(161, 19);
            this.lbltMainDeckCabin.TabIndex = 17;
            this.lbltMainDeckCabin.Text = "主板舱漏水报警：";
            // 
            // picMainDeckCabin
            // 
            this.picMainDeckCabin.BackColor = System.Drawing.Color.Red;
            this.picMainDeckCabin.Location = new System.Drawing.Point(496, 82);
            this.picMainDeckCabin.Name = "picMainDeckCabin";
            this.picMainDeckCabin.Size = new System.Drawing.Size(31, 31);
            this.picMainDeckCabin.TabIndex = 16;
            this.picMainDeckCabin.TabStop = false;
            // 
            // lbltControlModule
            // 
            this.lbltControlModule.AutoSize = true;
            this.lbltControlModule.Font = new System.Drawing.Font("宋体", 14F);
            this.lbltControlModule.Location = new System.Drawing.Point(329, 35);
            this.lbltControlModule.Name = "lbltControlModule";
            this.lbltControlModule.Size = new System.Drawing.Size(161, 19);
            this.lbltControlModule.TabIndex = 15;
            this.lbltControlModule.Text = "控制舱漏水报警：";
            // 
            // picControlModule
            // 
            this.picControlModule.BackColor = System.Drawing.Color.Red;
            this.picControlModule.Location = new System.Drawing.Point(496, 26);
            this.picControlModule.Name = "picControlModule";
            this.picControlModule.Size = new System.Drawing.Size(31, 31);
            this.picControlModule.TabIndex = 14;
            this.picControlModule.TabStop = false;
            // 
            // lblElectricity2
            // 
            this.lblElectricity2.AutoSize = true;
            this.lblElectricity2.Font = new System.Drawing.Font("宋体", 14F);
            this.lblElectricity2.Location = new System.Drawing.Point(158, 257);
            this.lblElectricity2.Name = "lblElectricity2";
            this.lblElectricity2.Size = new System.Drawing.Size(49, 19);
            this.lblElectricity2.TabIndex = 13;
            this.lblElectricity2.Text = "0.00";
            // 
            // lbltElectricity2
            // 
            this.lbltElectricity2.AutoSize = true;
            this.lbltElectricity2.Font = new System.Drawing.Font("宋体", 14F);
            this.lbltElectricity2.Location = new System.Drawing.Point(38, 257);
            this.lbltElectricity2.Name = "lbltElectricity2";
            this.lbltElectricity2.Size = new System.Drawing.Size(123, 19);
            this.lbltElectricity2.TabIndex = 12;
            this.lbltElectricity2.Text = "右路电流值：";
            // 
            // lblVoltage2
            // 
            this.lblVoltage2.AutoSize = true;
            this.lblVoltage2.Font = new System.Drawing.Font("宋体", 14F);
            this.lblVoltage2.Location = new System.Drawing.Point(158, 219);
            this.lblVoltage2.Name = "lblVoltage2";
            this.lblVoltage2.Size = new System.Drawing.Size(49, 19);
            this.lblVoltage2.TabIndex = 11;
            this.lblVoltage2.Text = "0.00";
            // 
            // lbltVoltage2
            // 
            this.lbltVoltage2.AutoSize = true;
            this.lbltVoltage2.Font = new System.Drawing.Font("宋体", 14F);
            this.lbltVoltage2.Location = new System.Drawing.Point(38, 219);
            this.lbltVoltage2.Name = "lbltVoltage2";
            this.lbltVoltage2.Size = new System.Drawing.Size(123, 19);
            this.lbltVoltage2.TabIndex = 10;
            this.lbltVoltage2.Text = "右路电压值：";
            // 
            // lblElectricity1
            // 
            this.lblElectricity1.AutoSize = true;
            this.lblElectricity1.Font = new System.Drawing.Font("宋体", 14F);
            this.lblElectricity1.Location = new System.Drawing.Point(158, 181);
            this.lblElectricity1.Name = "lblElectricity1";
            this.lblElectricity1.Size = new System.Drawing.Size(49, 19);
            this.lblElectricity1.TabIndex = 9;
            this.lblElectricity1.Text = "0.00";
            // 
            // lbltElectricity1
            // 
            this.lbltElectricity1.AutoSize = true;
            this.lbltElectricity1.Font = new System.Drawing.Font("宋体", 14F);
            this.lbltElectricity1.Location = new System.Drawing.Point(38, 181);
            this.lbltElectricity1.Name = "lbltElectricity1";
            this.lbltElectricity1.Size = new System.Drawing.Size(123, 19);
            this.lbltElectricity1.TabIndex = 8;
            this.lbltElectricity1.Text = "左路电流值：";
            // 
            // lblVoltage1
            // 
            this.lblVoltage1.AutoSize = true;
            this.lblVoltage1.Font = new System.Drawing.Font("宋体", 14F);
            this.lblVoltage1.Location = new System.Drawing.Point(158, 143);
            this.lblVoltage1.Name = "lblVoltage1";
            this.lblVoltage1.Size = new System.Drawing.Size(49, 19);
            this.lblVoltage1.TabIndex = 7;
            this.lblVoltage1.Text = "0.00";
            // 
            // lbltVoltage1
            // 
            this.lbltVoltage1.AutoSize = true;
            this.lbltVoltage1.Font = new System.Drawing.Font("宋体", 14F);
            this.lbltVoltage1.Location = new System.Drawing.Point(38, 143);
            this.lbltVoltage1.Name = "lbltVoltage1";
            this.lbltVoltage1.Size = new System.Drawing.Size(123, 19);
            this.lbltVoltage1.TabIndex = 6;
            this.lbltVoltage1.Text = "左路电压值：";
            // 
            // lblDepth
            // 
            this.lblDepth.AutoSize = true;
            this.lblDepth.Font = new System.Drawing.Font("宋体", 14F);
            this.lblDepth.Location = new System.Drawing.Point(158, 105);
            this.lblDepth.Name = "lblDepth";
            this.lblDepth.Size = new System.Drawing.Size(49, 19);
            this.lblDepth.TabIndex = 5;
            this.lblDepth.Text = "0.00";
            // 
            // lbltDepth
            // 
            this.lbltDepth.AutoSize = true;
            this.lbltDepth.Font = new System.Drawing.Font("宋体", 14F);
            this.lbltDepth.Location = new System.Drawing.Point(38, 105);
            this.lbltDepth.Name = "lbltDepth";
            this.lbltDepth.Size = new System.Drawing.Size(123, 19);
            this.lbltDepth.TabIndex = 4;
            this.lbltDepth.Text = "机器人深度：";
            // 
            // lblBoard
            // 
            this.lblBoard.AutoSize = true;
            this.lblBoard.Font = new System.Drawing.Font("宋体", 14F);
            this.lblBoard.Location = new System.Drawing.Point(158, 69);
            this.lblBoard.Name = "lblBoard";
            this.lblBoard.Size = new System.Drawing.Size(49, 19);
            this.lblBoard.TabIndex = 3;
            this.lblBoard.Text = "0.00";
            // 
            // lblTBoard
            // 
            this.lblTBoard.AutoSize = true;
            this.lblTBoard.Font = new System.Drawing.Font("宋体", 14F);
            this.lblTBoard.Location = new System.Drawing.Point(38, 69);
            this.lblTBoard.Name = "lblTBoard";
            this.lblTBoard.Size = new System.Drawing.Size(123, 19);
            this.lblTBoard.TabIndex = 2;
            this.lblTBoard.Text = "主板舱温度：";
            // 
            // lblTemperature
            // 
            this.lblTemperature.AutoSize = true;
            this.lblTemperature.Font = new System.Drawing.Font("宋体", 14F);
            this.lblTemperature.Location = new System.Drawing.Point(158, 34);
            this.lblTemperature.Name = "lblTemperature";
            this.lblTemperature.Size = new System.Drawing.Size(49, 19);
            this.lblTemperature.TabIndex = 1;
            this.lblTemperature.Text = "0.00";
            // 
            // lblttemperature
            // 
            this.lblttemperature.AutoSize = true;
            this.lblttemperature.Font = new System.Drawing.Font("宋体", 14F);
            this.lblttemperature.Location = new System.Drawing.Point(38, 34);
            this.lblttemperature.Name = "lblttemperature";
            this.lblttemperature.Size = new System.Drawing.Size(123, 19);
            this.lblttemperature.TabIndex = 0;
            this.lblttemperature.Text = "控制舱温度：";
            // 
            // skE
            // 
            this.skE.SerialNumber = "";
            this.skE.SkinFile = null;
            // 
            // tmrHandle
            // 
            this.tmrHandle.Tick += new System.EventHandler(this.tmrHandle_Tick);
            // 
            // tmrUI
            // 
            this.tmrUI.Tick += new System.EventHandler(this.tmrUI_Tick);
            // 
            // tmrSys
            // 
            this.tmrSys.Enabled = true;
            this.tmrSys.Tick += new System.EventHandler(this.tmrSys_Tick);
            // 
            // frmControl
            // 
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.None;
            this.ClientSize = new System.Drawing.Size(1031, 703);
            this.Controls.Add(this.panel1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MaximizeBox = false;
            this.Name = "frmControl";
            this.Text = "软体臂ROV操控台";
            this.Load += new System.EventHandler(this.frmControl_Load);
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            this.ssrStatusInfor.ResumeLayout(false);
            this.ssrStatusInfor.PerformLayout();
            this.grpOperation.ResumeLayout(false);
            this.grpOperation.PerformLayout();
            this.grpLEDAndSteeringEngine.ResumeLayout(false);
            this.grpLEDAndSteeringEngine.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trbSteering_Engine2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.trbSteering_Engine1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.trbLED)).EndInit();
            this.grpType.ResumeLayout(false);
            this.grpType.PerformLayout();
            this.grpTest.ResumeLayout(false);
            this.grpTest.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trbTranslation)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.trbRotate)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.trbAround)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.trbVertical)).EndInit();
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.grpRockerStateShow.ResumeLayout(false);
            this.grpRockerStateShow.PerformLayout();
            this.grpMachineStateShow.ResumeLayout(false);
            this.grpMachineStateShow.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.picMainDeckCabin)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.picControlModule)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.GroupBox grpMachineStateShow;
        private System.Windows.Forms.Label lblttemperature;
        private System.Windows.Forms.Label lblTemperature;
        private System.Windows.Forms.Label lblBoard;
        private System.Windows.Forms.Label lblTBoard;
        private System.Windows.Forms.Label lblElectricity1;
        private System.Windows.Forms.Label lbltElectricity1;
        private System.Windows.Forms.Label lblVoltage1;
        private System.Windows.Forms.Label lbltVoltage1;
        private System.Windows.Forms.Label lblDepth;
        private System.Windows.Forms.Label lbltDepth;
        private System.Windows.Forms.Label lblElectricity2;
        private System.Windows.Forms.Label lbltElectricity2;
        private System.Windows.Forms.Label lblVoltage2;
        private System.Windows.Forms.Label lbltVoltage2;
        private System.Windows.Forms.Label lbltControlModule;
        private System.Windows.Forms.PictureBox picControlModule;
        private System.Windows.Forms.Label lbltMainDeckCabin;
        private System.Windows.Forms.PictureBox picMainDeckCabin;
        private HeadingIndicator.HeadingIndicator hiIndicator;
        private System.Windows.Forms.GroupBox grpRockerStateShow;
        private System.Windows.Forms.Label lblFloating;
        private System.Windows.Forms.Label lblDive;
        private System.Windows.Forms.VScrollBar vscY;
        private System.Windows.Forms.Label lblLeftShift;
        private System.Windows.Forms.HScrollBar hscX;
        private System.Windows.Forms.Label lblRightShift;
        private System.Windows.Forms.Label lblTurnRight;
        private System.Windows.Forms.Label lblTurnLeft;
        private System.Windows.Forms.HScrollBar hscZ;
        private System.Windows.Forms.VScrollBar vscR;
        private System.Windows.Forms.Label lblBackOff;
        private System.Windows.Forms.Label lblForward;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Label lbltIP;
        private System.Windows.Forms.TextBox txtPort;
        private System.Windows.Forms.Label lblPort;
        private System.Windows.Forms.TextBox txtIP;
        private System.Windows.Forms.Button btnConSocket;
        private System.Windows.Forms.GroupBox grpTest;
        private Sunisoft.IrisSkin.SkinEngine skE;
        private System.Windows.Forms.GroupBox grpType;
        private System.Windows.Forms.RadioButton rdoCloseLoop;
        private System.Windows.Forms.RadioButton rdoOpenLoop;
        private System.Windows.Forms.Button btnCloseRotate;
        private System.Windows.Forms.TrackBar trbRotate;
        private System.Windows.Forms.Label lblRotate;
        private System.Windows.Forms.Button btnCloseAround;
        private System.Windows.Forms.TrackBar trbAround;
        private System.Windows.Forms.Label lblAround;
        private System.Windows.Forms.Button btnCloseVertical;
        private System.Windows.Forms.TrackBar trbVertical;
        private System.Windows.Forms.Label lblVertical;
        private System.Windows.Forms.GroupBox grpLEDAndSteeringEngine;
        private System.Windows.Forms.Button btnClose_Steering_Engine2;
        private System.Windows.Forms.TrackBar trbSteering_Engine2;
        private System.Windows.Forms.Label lblSteering_Engine2;
        private System.Windows.Forms.Button btnClose_Steering_Engine1;
        private System.Windows.Forms.TrackBar trbSteering_Engine1;
        private System.Windows.Forms.Label lblSteering_Engine1;
        private System.Windows.Forms.Button btnCloseLED;
        private System.Windows.Forms.TrackBar trbLED;
        private System.Windows.Forms.Label lblLED;
        private System.Windows.Forms.GroupBox grpOperation;
        private System.Windows.Forms.ComboBox cmbHandle;
        private System.Windows.Forms.Button btnConHandle;
        private System.Windows.Forms.Label lblHandle;
        private System.Windows.Forms.Timer tmrHandle;
        private System.Windows.Forms.Timer tmrUI;
        private System.Windows.Forms.StatusStrip ssrStatusInfor;
        private System.Windows.Forms.ToolStripStatusLabel toolSStLbl;
        private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel1;
        private System.Windows.Forms.ToolStripStatusLabel ttrType;
        private System.Windows.Forms.ToolStripStatusLabel tssJoyCon;
        private System.Windows.Forms.ToolStripStatusLabel tssConTerminal;
        private System.Windows.Forms.Button btnCloseTranslation;
        private System.Windows.Forms.TrackBar trbTranslation;
        private System.Windows.Forms.Label lblTranslation;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.ToolStripStatusLabel tssTime;
        private System.Windows.Forms.Timer tmrSys;
    }
}